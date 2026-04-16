import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import math
# import cv2
# import numpy as np
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
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
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

        # TODO: 請替換為你的機器人 IP 與帳密
        # self.robot = sdk.create_robot("10.0.0.3")
        self.robot = sdk.create_robot("192.168.80.3")
        self.robot.authenticate("admin", "eqyqp33u8i74")
        self.robot.time_sync.wait_for_sync()
        self.ncb_client = self.robot.ensure_client(NetworkComputeBridgeClient.default_service_name)
        
        # --- 3. 任務參數 ---
        self.ml_service = "fetch-server"       # TODO: 填入你的服務名稱
        self.model_name = "best.engine"        # TODO: 填入你的模型名稱
        self.target_label = "Bottle_and_Can"   # TODO: 你要找的標籤
        self.min_confidence = 0.5

        # 去重門檻：5 cm
        self.duplicate_threshold = 0.05
        
        # 啟動主迴圈計時器
        self.timer = self.create_timer(0.5, self.fetch_loop)
        self.is_fetching = False
        self.is_approaching = False
        self.move_msg = Twist()

        # ----------------------- walk --------------------------------------------------
        # 訂閱 nav 速度指令
        self.nav_sub = self.create_subscription(Twist, 'cmd_vel_nav', self.nav_callback, 10)
        # 發布最終給 Spot 的控制速度
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 狀態變數
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vrot = 0.0

        # ----------------------- detection / target list -------------------------------
        self.detection_round = 0
        self.detected_objects = []   # 本輪清單
        self.target_list = []        # 全局清單
        self.next_target_id = 0

        # ----------------------- RViz publisher ---------------------------------------
        self.marker_pub = self.create_publisher(MarkerArray, 'detected_target_markers', 10)

        # ----------------------- TF ---------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def nav_callback(self, msg):
        """
        核心邏輯：如果沒在夾東西，就把導航的速度直接『搬運』給 Spot。
        如果正在夾東西，則不轉發（或者發布 0,0,0 確保靜止）。
        """
        if not self.is_fetching and not self.is_approaching:
            self.cmd_vel_pub.publish(msg)
        elif self.is_approaching:
            self.cmd_vel_pub.publish(self.move_msg)
        else:
            # 如果正在處理物品，我們不主動發布 0，
            # 讓 fetch_loop 控制速度就好，避免頻率衝突
            pass
    
    def fetch_loop(self):
        if self.is_fetching:
            return
        
        self.get_logger().info('正在透過 NCS 搜尋物體...')

        # 1. 掃描所有看到的目標
        self.detection_obj_and_img(
            ['frontleft_fisheye_image', 'frontright_fisheye_image']
        )

        # 2. 更新全局 target_list（5 cm 去重）
        self.update_target_list()

        # 3. 發布到 RViz
        self.publish_target_markers()

        # 4. 若沒有任何目標，恢復巡邏 (走coverage_path)
        if len(self.target_list) == 0:
            if self.is_approaching:
                self.get_logger().info("target_list 為空，恢復巡邏模式...")
                self.is_approaching = False
            return

        # 5. 從 target_list 中找最近目標
        nearest_target, nearest_pose_in_body, nearest_distance = self.find_nearest_target()

        #防止TF轉換失敗回傳 None 的情況導致後續程式崩潰
        if nearest_target is None or nearest_pose_in_body is None or nearest_distance is None:
            if self.is_approaching:
                self.get_logger().info("找不到有效最近目標，恢復巡邏模式...")
                self.is_approaching = False
            return
        
        self.get_logger().info(
            f"目前最近目標: {nearest_target['id']}，距離: {nearest_distance:.2f}m"
        )

        # 6. 若最近目標還大於 1.5m，就接近最近目標
        if nearest_distance > 1.5:
            self.is_approaching = True

            tx = nearest_pose_in_body.pose.position.x
            ty = nearest_pose_in_body.pose.position.y
            angle_to_target = math.atan2(ty, tx)

            self.get_logger().info(
                f"目前最近目標: {nearest_target['id']}，"
                f"body x={tx:.2f}, y={ty:.2f}, 距離={nearest_distance:.2f}m"
            )

            self.move_msg.linear.x = 0.3
            self.move_msg.angular.z = angle_to_target * 0.5
            return

        # 7. 若最近目標小於等於 1.5m，進入夾取模式
        self.get_logger().info("最近目標已進入 1.5m 範圍，開始進入夾取模式...")
        self.is_approaching = False

        target_obj, image_full, vision_tform_obj = self.get_obj_and_img(
            ['frontleft_fisheye_image', 'frontright_fisheye_image']
        )

        if target_obj is None or vision_tform_obj is None:
            self.get_logger().warn("進入夾取模式，但 get_obj_and_img() 沒有取得有效目標")
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
            distance = math.sqrt(tx**2 + ty**2)

            self.get_logger().info(f"相對座標: 前方={tx:.2f}m, 左方={ty:.2f}m")
            self.get_logger().info(f"夾取前確認距離: {distance:.2f}m")

        except Exception as e:
            self.get_logger().error(f"夾取前座標轉換失敗: {e}")
            return

        self.get_logger().info("已抵達目標範圍，開始夾取程序...")
        self.is_fetching = True

        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        # 計算像素中心
        center_px_x, center_px_y = self.find_center_px(target_obj.image_properties.coordinates)

        # 構造 SDK 夾取指令
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

        # --- 將 SDK 夾取指令轉換並傳送給 ROS 2 ---
        self.send_ros2_manipulation_goal(manip_request)

    def send_cmd_blocking(self, sdk_cmd, label):
        """同步阻塞式發送：確保前一個動作完成才回傳"""
        if not self.robot_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('無法連接到 Action Server，請確認 spot_driver 狀態')
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

    # ---------------------------------------------------------
    # 單目標夾取用函式(只處理1.5m內的目標並且回傳最近的的目標)
    # ---------------------------------------------------------
    def get_obj_and_img(self, image_sources):
        best_obj = None
        best_image_response = None
        best_vision_tform_obj = None
        nearest_distance = math.inf

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

                try:
                    vision_tform_body = frame_helpers.get_a_tform_b(
                        resp.image_response.shot.transforms_snapshot,
                        frame_helpers.VISION_FRAME_NAME,
                        frame_helpers.BODY_FRAME_NAME
                    )

                    body_tform_obj = vision_tform_body.inverse() * vision_tform_obj

                    tx = body_tform_obj.x
                    ty = body_tform_obj.y
                    distance = math.sqrt(tx ** 2 + ty ** 2)
                except Exception as e:
                    self.get_logger().warn(f'距離計算失敗 ({source}): {e}')
                    continue

                # 只處理 1.5m 內的目標
                if distance > 1.5:
                    continue

                # 取最近的目標回傳
                if distance < nearest_distance:
                    nearest_distance = distance
                    best_obj = obj
                    best_image_response = resp.image_response
                    best_vision_tform_obj = vision_tform_obj

        if best_obj is not None:
            return best_obj, best_image_response, best_vision_tform_obj

        return None, None, None

    # ---------------------------------------------------------
    # 多目標掃描函式
    # ---------------------------------------------------------
    def detection_obj_and_img(self, image_sources):
        # 每輪清單初始化
        self.detection_round += 1
        round_id = self.detection_round

        # 清空本輪候選物件
        self.detected_objects = []

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
                })

    # ---------------------------------------------------------
    # 更新全局 target_list，5 cm 去重
    # ---------------------------------------------------------
    def update_target_list(self):
        for detected in self.detected_objects:
            new_tform = detected["vision_tform_obj"]
            matched_index = None

            for i, target in enumerate(self.target_list):
                old_tform = target["vision_tform_obj"]

                dx = new_tform.x - old_tform.x
                dy = new_tform.y - old_tform.y
                dz = new_tform.z - old_tform.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                if distance <= self.duplicate_threshold:
                    matched_index = i
                    break

            if matched_index is not None:
                # 同一物件：更新座標，保留原本全局 ID
                self.target_list[matched_index]["obj"] = detected["obj"]
                self.target_list[matched_index]["vision_tform_obj"] = detected["vision_tform_obj"]
            else:
                target_id = f"target_{self.next_target_id}"
                self.next_target_id += 1

                self.target_list.append({
                    "obj": detected["obj"],
                    "id": target_id,
                    "vision_tform_obj": detected["vision_tform_obj"],
                })
    # ---------------------------------------------------------
    # 搜尋全局 target_list 中最近的目標 有問題
    # ---------------------------------------------------------
    def find_nearest_target(self):
        if len(self.target_list) == 0:
            return None, None, None

        nearest_target = None
        nearest_pose_in_body = None
        nearest_distance = math.inf

        for target in self.target_list:
            tform = target["vision_tform_obj"]

            try:
                pose_in_vision = PoseStamped()
                pose_in_vision.header.stamp = self.get_clock().now().to_msg()
                pose_in_vision.header.frame_id = frame_helpers.VISION_FRAME_NAME

                pose_in_vision.pose.position.x = float(tform.x)
                pose_in_vision.pose.position.y = float(tform.y)
                pose_in_vision.pose.position.z = float(tform.z)

                pose_in_vision.pose.orientation.x = 0.0
                pose_in_vision.pose.orientation.y = 0.0
                pose_in_vision.pose.orientation.z = 0.0
                pose_in_vision.pose.orientation.w = 1.0

                pose_in_body = self.tf_buffer.transform(
                    pose_in_vision,
                    frame_helpers.BODY_FRAME_NAME,
                    timeout=Duration(seconds=0.2)
                )

                x = pose_in_body.pose.position.x
                y = pose_in_body.pose.position.y

                # 只算平面距離
                distance = math.sqrt(x * x + y * y)

            except Exception as e:
                self.get_logger().warn(f"目標 {target['id']} TF 轉換失敗: {e}")
                continue

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_target = target
                nearest_pose_in_body = pose_in_body

        if nearest_target is None:
            return None, None, None

        return nearest_target, nearest_pose_in_body, nearest_distance
    # ---------------------------------------------------------
    # 發布 RViz marker
    # ---------------------------------------------------------
    def publish_target_markers(self):
        marker_array = MarkerArray()

        # 先刪除舊 marker
        delete_marker = Marker()
        delete_marker.header.frame_id = frame_helpers.VISION_FRAME_NAME
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i, target in enumerate(self.target_list):
            tform = target["vision_tform_obj"]

            # 球體 marker
            marker = Marker()
            marker.header.frame_id = frame_helpers.VISION_FRAME_NAME
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "detected_targets"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(tform.x)
            marker.pose.position.y = float(tform.y)
            marker.pose.position.z = float(tform.z)

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08

            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker.lifetime.sec = 0
            marker_array.markers.append(marker)

            # 文字 marker
            text_marker = Marker()
            text_marker.header.frame_id = frame_helpers.VISION_FRAME_NAME
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "detected_target_labels"
            text_marker.id = 1000 + i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(tform.x)
            text_marker.pose.position.y = float(tform.y)
            text_marker.pose.position.z = float(tform.z) + 0.12

            text_marker.pose.orientation.x = 0.0
            text_marker.pose.orientation.y = 0.0
            text_marker.pose.orientation.z = 0.0
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.08

            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0

            text_marker.text = target["id"]
            text_marker.lifetime.sec = 0
            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)

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
