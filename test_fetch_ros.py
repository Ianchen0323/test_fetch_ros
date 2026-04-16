import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import math
import cv2
import numpy as np
from google.protobuf import wrappers_pb2

# Boston Dynamics SDK (僅用於視覺與資料處理，不搶 Lease)
import bosdyn.client
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2, network_compute_bridge_pb2
from bosdyn.client import frame_helpers
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.client.robot_command import RobotCommandBuilder

# ROS 2 轉換與 Action 訊息
from bosdyn_msgs.conversions import convert
from spot_msgs.action import Manipulation
from spot_msgs.action import RobotCommand

import threading
import time

# ros2
from geometry_msgs.msg import Twist

# rviz 可視化
from visualization_msgs.msg import Marker, MarkerArray


class SpotFetchROS2Node(Node):
    def __init__(self):
        super().__init__('spot_fetch_ros2_node')

        # --- 1. ROS 2 設定 ---
        self.manip_client = ActionClient(self, Manipulation, '/manipulation')
        self.robot_client = ActionClient(self, RobotCommand, '/robot_command')

        # --- 2. SDK 設定 (不要求 Lease) ---
        self.get_logger().info('正在設定robot...')
        self.get_logger().info("正在連線至 Spot SDK (僅讀取影像/NCS)...")
        sdk = bosdyn.client.create_standard_sdk('SpotFetchROS2')
        sdk.register_service_client(NetworkComputeBridgeClient)

        self.robot = sdk.create_robot("192.168.80.3")
        self.robot.authenticate("admin", "eqyqp33u8i74")
        self.robot.time_sync.wait_for_sync()
        self.ncb_client = self.robot.ensure_client(NetworkComputeBridgeClient.default_service_name)

        # --- 3. 任務參數 ---
        self.ml_service = "fetch-server"
        self.model_name = "best.engine"
        self.target_label = "Bottle_and_Can"
        self.min_confidence = 0.5

        # --- 4. 偵測 / 目標清單 ---
        self.detected_objects = []   # 本輪偵測結果
        self.targets_list = []       # 跨輪待撿清單
        self.detection_round = 0

        # --- 5. RViz 可視化 ---
        # 直接使用 vision frame 顯示，不再做 vision -> map 的 TF 轉換
        self.targets_marker_pub = self.create_publisher(
            MarkerArray,
            '/targets_markers',
            10
        )

        # 啟動主迴圈計時器
        self.timer = self.create_timer(0.5, self.fetch_loop)
        self.is_fetching = False
        self.is_approaching = False
        self.move_msg = Twist()

        # -----------------------walk--------------------------------------------------
        self.nav_sub = self.create_subscription(Twist, 'cmd_vel_nav', self.nav_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 狀態變數
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vrot = 0.0

    def nav_callback(self, msg):
        if not self.is_fetching and not self.is_approaching:
            self.cmd_vel_pub.publish(msg)
        elif self.is_approaching:
            self.cmd_vel_pub.publish(self.move_msg)
        else:
            pass

    def pose_distance(self, pose1, pose2):
        dx = pose1.x - pose2.x
        dy = pose1.y - pose2.y
        dz = pose1.z - pose2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def update_targets_list(self, merge_dist=0.05):
        # 全域清單 targets_list 更新
        for det in self.detected_objects:
            matched = False

            for target in self.targets_list:
                d = self.pose_distance(det["vision_tform_obj"], target["vision_tform_obj"])
                if d < merge_dist:
                    target["vision_tform_obj"] = det["vision_tform_obj"]
                    target["confidence"] = det["confidence"]
                    matched = True
                    break

            if not matched:
                self.targets_list.append({
                    "id": det["id"],
                    "vision_tform_obj": det["vision_tform_obj"],
                    "confidence": det["confidence"],
                    "status": "pending",
                })

    def publish_targets_markers(self):
        # 直接將 targets_list 以 vision frame 可視化到 RViz
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        marker_id = 0

        for target in self.targets_list:
            vision_tform_obj = target["vision_tform_obj"]

            # 球體 marker
            sphere_marker = Marker()
            sphere_marker.header.frame_id = 'vision'
            sphere_marker.header.stamp = now
            sphere_marker.ns = 'targets'
            sphere_marker.id = marker_id
            marker_id += 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD

            sphere_marker.pose.position.x = vision_tform_obj.x
            sphere_marker.pose.position.y = vision_tform_obj.y
            sphere_marker.pose.position.z = vision_tform_obj.z
            sphere_marker.pose.orientation.x = 0.0
            sphere_marker.pose.orientation.y = 0.0
            sphere_marker.pose.orientation.z = 0.0
            sphere_marker.pose.orientation.w = 1.0

            sphere_marker.scale.x = 0.15
            sphere_marker.scale.y = 0.15
            sphere_marker.scale.z = 0.15

            if target["status"] == "pending":
                sphere_marker.color.r = 1.0
                sphere_marker.color.g = 1.0
                sphere_marker.color.b = 0.0
                sphere_marker.color.a = 1.0
            elif target["status"] == "active":
                sphere_marker.color.r = 0.0
                sphere_marker.color.g = 1.0
                sphere_marker.color.b = 0.0
                sphere_marker.color.a = 1.0
            else:
                sphere_marker.color.r = 0.5
                sphere_marker.color.g = 0.5
                sphere_marker.color.b = 0.5
                sphere_marker.color.a = 1.0

            marker_array.markers.append(sphere_marker)

            # 文字 marker
            text_marker = Marker()
            text_marker.header.frame_id = 'vision'
            text_marker.header.stamp = now
            text_marker.ns = 'targets_text'
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = vision_tform_obj.x
            text_marker.pose.position.y = vision_tform_obj.y
            text_marker.pose.position.z = vision_tform_obj.z + 0.2
            text_marker.pose.orientation.x = 0.0
            text_marker.pose.orientation.y = 0.0
            text_marker.pose.orientation.z = 0.0
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.12
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f'{target["id"]} ({target["confidence"]:.2f})'

            marker_array.markers.append(text_marker)

        self.targets_marker_pub.publish(marker_array)

    def fetch_loop(self):
        if self.is_fetching:
            return

        self.get_logger().info('正在透過 NCS 搜尋物體...')

        # 取得影像與辨識結果
        target_obj, image_full, vision_tform_obj = self.get_obj_and_img(
            ['frontleft_fisheye_image', 'frontright_fisheye_image']
        )

        # 每輪都把本輪偵測結果更新進待撿清單
        self.update_targets_list()

        # 每輪都把 targets_list 直接用 vision frame 發布到 RViz
        self.publish_targets_markers()

        if target_obj is None or vision_tform_obj is None:
            if self.is_approaching:
                self.get_logger().info("目標遺失，恢復巡邏模式...")
                self.is_approaching = False
            return

        try:
            vision_tform_body = frame_helpers.get_a_tform_b(
                image_full.shot.transforms_snapshot,
                frame_helpers.VISION_FRAME_NAME,
                frame_helpers.BODY_FRAME_NAME
            )

            body_tform_obj = vision_tform_body.inverse() * vision_tform_obj

            tx = body_tform_obj.x
            ty = body_tform_obj.y
            distance = math.sqrt(tx ** 2 + ty ** 2)
            angle_to_target = math.atan2(ty, tx)

            self.get_logger().info(f"相對座標: 前方={tx:.2f}m, 左方={ty:.2f}m")

        except Exception as e:
            self.get_logger().error(f"座標轉換失敗: {e}")
            return

        self.get_logger().info(f'偵測到目標，距離: {distance:.2f} 公尺')

        if distance > 8.0:
            self.get_logger().info("目標太遠，忽略中...")
            return

        elif 2.0 < distance <= 8.0:
            self.get_logger().info(f"接近目標中... (距離 {distance:.2f}m)")
            self.is_approaching = True

            self.move_msg.linear.x = 0.3
            self.move_msg.angular.z = angle_to_target * 0.5
            return

        else:
            self.get_logger().info("已抵達目標範圍，開始夾取程序...")
            self.is_approaching = False
            self.is_fetching = True

            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)

        center_px_x, center_px_y = self.find_center_px(target_obj.image_properties.coordinates)

        pick_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image_full.shot.transforms_snapshot,
            frame_name_image_sensor=image_full.shot.frame_name_image_sensor,
            camera_model=image_full.source.pinhole
        )
        grasp.grasp_params.grasp_palm_to_fingertip = 0.6
        grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME

        manip_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp
        )

        self.send_ros2_manipulation_goal(manip_request)

    def send_cmd_blocking(self, sdk_cmd, label):
        if not self.robot_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'無法連接到 Action Server，請確認 spot_driver 狀態')
            return False

        goal_msg = RobotCommand.Goal()
        convert(sdk_cmd, goal_msg.command)

        self.get_logger().info(f'▶執行步驟: {label}')

        send_goal_future = self.robot_client.send_goal_async(goal_msg)
        while rclpy.ok() and not send_goal_future.done():
            time.sleep(0.1)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'{label} 被機器人拒絕')
            return False

        result_future = goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            time.sleep(0.1)

        self.get_logger().info(f'{label} 完成')
        return True

    def send_ros2_manipulation_goal(self, manip_request):
        if not self.manip_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("找不到 Manipulation Action Server!")
            self.is_fetching = False
            return

        goal_msg = Manipulation.Goal()
        convert(manip_request, goal_msg.command)

        self.get_logger().info('正在發送 ROS 2 夾取指令...')
        send_goal_future = self.manip_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('夾取請求被拒絕！')
            self.is_fetching = False
            return

        self.get_logger().info('夾取請求已接受，執行中...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        if status == 4:
            self.get_logger().info('✅ 夾取動作確認成功！')
            action_thread = threading.Thread(target=self.post_grasp_sequence)
            action_thread.start()
        else:
            self.get_logger().error(f'❌ 夾取失敗，狀態碼: {status}')
            self.is_fetching = False

    def post_grasp_sequence(self):
        self.send_cmd_blocking(RobotCommandBuilder.arm_ready_command(), "手臂預備 (Ready)")

        try:
            poses = [
                (0.25, 0.2, 0.6, "安全點 1: 側上方"),
                (0.0, 0.0, 0.7, "安全點 2: 正上方"),
                (-0.2, 0.0, 0.6, "安全點 3: 背部後方")
            ]

            for x, y, z, label in poses:
                cmd = RobotCommandBuilder.arm_pose_command(
                    x, y, z, 0.707, 0, 0.707, 0,
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME,
                    seconds=1.5
                )
                self.send_cmd_blocking(cmd, label)

            self.send_cmd_blocking(
                RobotCommandBuilder.claw_gripper_open_fraction_command(1.0),
                "開啟夾爪"
            )
            time.sleep(1.0)

            self.send_cmd_blocking(
                RobotCommandBuilder.arm_stow_command(),
                "手臂收納 (Stow)"
            )

            self.send_cmd_blocking(
                RobotCommandBuilder.claw_gripper_open_fraction_command(0.0),
                "關閉夾爪"
            )
            time.sleep(1.0)

        except Exception as e:
            self.get_logger().error(f"執行回收動作時出錯: {e}")
        finally:
            self.is_fetching = False

    def get_obj_and_img(self, image_sources):
        # 每輪清單初始化
        self.detection_round += 1
        round_id = self.detection_round

        # 清空本輪候選物件
        self.detected_objects = []

        best_obj = None
        best_image_response = None
        highest_conf = 0.0
        best_vision_tform_obj = None

        # 同一輪內的物件編號
        obj_index_in_round = 0

        for source in image_sources:
            img_src = network_compute_bridge_pb2.ImageSourceAndService(
                image_source=source
            )
            input_data = network_compute_bridge_pb2.NetworkComputeInputData(
                image_source_and_service=img_src,
                model_name=self.model_name,
                min_confidence=self.min_confidence,
                rotate_image=network_compute_bridge_pb2.NetworkComputeInputData.ROTATE_IMAGE_ALIGN_HORIZONTAL
            )
            server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(
                service_name=self.ml_service
            )
            process_img_req = network_compute_bridge_pb2.NetworkComputeRequest(
                input_data=input_data,
                server_config=server_data
            )

            try:
                resp = self.ncb_client.network_compute_bridge_command(process_img_req)
            except Exception as e:
                self.get_logger().error(f'NCS 連線錯誤 ({source}): {e}')
                continue

            if len(resp.object_in_image) == 0:
                continue

            for obj in resp.object_in_image:
                obj_label = obj.name.split('_label_')[-1]
                if obj_label != self.target_label:
                    continue

                conf_msg = wrappers_pb2.FloatValue()
                obj.additional_properties.Unpack(conf_msg)
                conf = conf_msg.value

                try:
                    vision_tform_obj = frame_helpers.get_a_tform_b(
                        obj.transforms_snapshot,
                        frame_helpers.VISION_FRAME_NAME,
                        obj.image_properties.frame_name_image_coordinates
                    )
                except Exception:
                    vision_tform_obj = None

                if vision_tform_obj is None:
                    continue

                suffix = chr(ord('a') + obj_index_in_round)
                obj_id = f"{round_id}{suffix}"
                obj_index_in_round += 1

                self.detected_objects.append({
                    "obj": obj,
                    "id": obj_id,
                    "vision_tform_obj": vision_tform_obj,
                    "confidence": conf,
                })

                if conf > highest_conf:
                    highest_conf = conf
                    best_obj = obj
                    best_image_response = resp.image_response
                    best_vision_tform_obj = vision_tform_obj

        if best_obj is not None:
            return best_obj, best_image_response, best_vision_tform_obj

        return None, None, None

    def find_center_px(self, polygon):
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        for vert in polygon.vertexes:
            if vert.x < min_x:
                min_x = vert.x
            if vert.y < min_y:
                min_y = vert.y
            if vert.x > max_x:
                max_x = vert.x
            if vert.y > max_y:
                max_y = vert.y
        x = math.fabs(max_x - min_x) / 2.0 + min_x
        y = math.fabs(max_y - min_y) / 2.0 + min_y
        return (x, y)


def main(args=None):
    rclpy.init(args=args)
    node = SpotFetchROS2Node()

    # 使用多執行緒執行器
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('正在關閉節點...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()