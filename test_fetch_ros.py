import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import math
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
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from geometry_msgs.msg import Twist

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

# ----------------------- status 定義 -------------------------------------------
STATUS_DETECTED    = "detected"
STATUS_PENDING     = "pending"
STATUS_APPROACHING = "approaching"
STATUS_GRASPING    = "grasping"
STATUS_GRASPED     = "grasped"
STATUS_UNHANDLED   = "unhandled"


class SpotFetchROS2Node(Node):
    def __init__(self):
        super().__init__('spot_fetch_ros2_node')

        self.action_group = ReentrantCallbackGroup()
        self._manip_result_future = None

        # --- 1. ROS 2 設定 ---
        # self.manip_client = ActionClient(self, Manipulation, '/manipulation')
        self.manip_client = ActionClient(
            self, 
            Manipulation, 
            '/manipulation',
            callback_group=self.action_group  # 明確指定
        )
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
        self.duplicate_threshold = 0.2

        # tf失敗後的狀態維持時間
        self.tf_cache_timeout_sec = 1.0
        
        # 啟動主迴圈計時器
        self.is_fetching = False
        self.is_approaching = False
        self.move_msg = Twist()

        # 建立一個並行群組
        self.nav_group = ReentrantCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()
        
        # ----------------------- communication --------------------------------------------------
        
        # 1. 導航訂閱：使用並行群組
        self.nav_sub = self.create_subscription(
            Twist, 'cmd_vel_nav', self.nav_callback, 10,
            callback_group=self.nav_group
        )
        
        # 2. NCS 偵測計時器：使用並行群組
        self.timer = self.create_timer(
            0.1, self.fetch_loop,
            callback_group=self.timer_group
        )
        
        # # 訂閱 nav 速度指令
        # self.nav_sub = self.create_subscription(Twist, 'cmd_vel_nav', self.nav_callback, 10)
        # 發布最終給 Spot 的控制速度
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)


        # ----------------------- detection / target list -------------------------------
        self.detection_round = 0
        self.detected_objects = []   # 本輪清單
        self.target_list = []        # 全局清單
        self.next_target_id = 0
        self.current_grasp_target_id = None

        # ----------------------- TF ---------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # -------------------------------------------------------------------------------

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

        target_obj, image_full, vision_tform_obj, target_id = self.detection_obj_and_img(
            ['frontleft_fisheye_image', 'frontright_fisheye_image']
        )

        self.get_logger().info("=== target_list ===")
        for target in self.target_list:
            self.get_logger().info(f"{target['id']}, {target['status']}")


        if len(self.target_list) == 0:
            if self.is_approaching:
                self.get_logger().info("target_list 為空，恢復巡邏模式...")
                self.is_approaching = False
            return

        if target_id is not None:
            self.get_logger().info("已有 pending 目標，開始進入夾取模式...")
            self.is_approaching = False

            if target_obj is None or vision_tform_obj is None:
                self.get_logger().warn("進入夾取模式，但 detection_obj_and_img() 沒有取得有效目標")
                return

            self.get_logger().info("已抵達目標範圍，開始夾取程序...")
            self.is_fetching = True
            self.current_grasp_target_id = target_id

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
            return

        approaching_target = None
        approaching_pose_in_body = None
        approaching_distance = None
        now_sec = self.get_clock().now().nanoseconds / 1e9

        for target in self.target_list:
            if target.get("status") != STATUS_APPROACHING:
                continue

            tform = target["vision_tform_obj"]
            pose_in_body = None
            distance = None

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

                # TF 成功：更新 pose_in_body / last_time
                target["pose_in_body"] = pose_in_body
                target["last_time"] = now_sec

                x = pose_in_body.pose.position.x
                y = pose_in_body.pose.position.y
                distance = math.sqrt(x * x + y * y)

            except Exception as e:
                cached_pose = target.get("pose_in_body", None)
                last_time = target.get("last_time", None)

                # TF 失敗：若舊 pose_in_body 還夠新，就沿用做移動
                if cached_pose is not None and last_time is not None and (now_sec - last_time) < self.tf_cache_timeout_sec:
                    pose_in_body = cached_pose
                    x = pose_in_body.pose.position.x
                    y = pose_in_body.pose.position.y
                    distance = math.sqrt(x * x + y * y)
                else:
                    continue

            approaching_target = target
            approaching_pose_in_body = pose_in_body
            approaching_distance = distance
            break

        if approaching_target is not None:
            self.is_approaching = True

            tx = approaching_pose_in_body.pose.position.x
            ty = approaching_pose_in_body.pose.position.y
            angle_to_target = math.atan2(ty, tx)

            self.get_logger().info(
                f"目前接近目標: {approaching_target['id']}，"
                f"body x={tx:.2f}, y={ty:.2f}, 距離={approaching_distance:.2f}m"
            )

            self.move_msg = Twist()

            if abs(angle_to_target) > 1.2:
                self.move_msg.linear.x = 0.0
                self.move_msg.angular.z = angle_to_target * 0.6
            elif abs(angle_to_target) > 0.5:
                self.move_msg.linear.x = 0.12
                self.move_msg.angular.z = angle_to_target * 0.6
            else:
                self.move_msg.linear.x = 0.3
                self.move_msg.angular.z = angle_to_target * 0.5

            self.cmd_vel_pub.publish(self.move_msg)
            return

        if self.is_approaching:
            self.get_logger().info("目前沒有 approaching / pending 目標，恢復巡邏模式...")
            self.is_approaching = False

    def send_cmd_async(self, sdk_cmd, label):
        """非同步發送：發完指令就直接回傳，不等待結果"""
        goal_msg = RobotCommand.Goal()
        convert(sdk_cmd, goal_msg.command)
        self.get_logger().info(f'發布指令: {label}')
        self.robot_client.send_goal_async(goal_msg) # 不加 callback，不等待
    
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

            target = self.get_target_by_id(self.current_grasp_target_id)
            if target is not None and target.get("status") != STATUS_GRASPED:
                target["status"] = STATUS_PENDING

            self.current_grasp_target_id = None
            self.is_fetching = False
            return

        # action server 已接受，這時才設成 grasping
        target = self.get_target_by_id(self.current_grasp_target_id)
        if target is not None:
            target["status"] = STATUS_GRASPING

        self.get_logger().info('夾取請求已接受，執行中...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self._manip_result_future = None
        status = future.result().status
        target = self.get_target_by_id(self.current_grasp_target_id)

        if status == 4:
            self.get_logger().info('✅ 夾取動作確認成功！')

            if target is not None:
                target["status"] = STATUS_GRASPED
                target["fail_count"] = 0

            action_thread = threading.Thread(target=self.post_grasp_sequence)
            action_thread.start()

        elif status == 6:  # ABORTED
            self.get_logger().error('❌ 夾取被中止 (ABORTED)。通常是影像跟丟或位置不好，準備重試...')
            self.handle_grasp_failure(target)

        else:
            self.get_logger().error(f'❌ 夾取失敗，狀態碼: {status}')
            self.handle_grasp_failure(target)
        
    def post_grasp_sequence(self):
        self.send_cmd_blocking(RobotCommandBuilder.arm_ready_command(), "手臂預備 (Ready)")
        
        try:
            final_joints = [3.103, -1.219, 0.732, 0.013, 1.826, 2.877]
            
            # --- 步驟 1: 高位轉向後方 ---
            # 我們保留 sh0 (3.103)，但將 sh1 設為較高的角度 (例如 -0.5 或 -0.8)
            # 這樣手臂會「舉著」轉到後面，不會掃到背上的設備
            high_rotate_joints = [2.38, -1.719, 0.8, 0.0, 1.826, 2.877] 
            
            self.get_logger().info("執行步驟 1: 高位旋轉至後方...")
            cmd1 = RobotCommandBuilder.arm_joint_command(*high_rotate_joints)
            # self.send_cmd_blocking(cmd1, "高位旋轉")
            self.send_cmd_async(cmd1, "高位旋轉")
            time.sleep(1.2)
            # --- 步驟 2: 下壓至最終目標點 ---
            # 此時第一軸已經到位，垂直降落即可
            self.get_logger().info("執行步驟 2: 下壓至背後安全點...")
            cmd2 = RobotCommandBuilder.arm_joint_command(*final_joints, max_vel=20.0, 
                max_accel=5.0)
            self.send_cmd_blocking(cmd2, "垂直下壓到位")
            
            
            self.send_cmd_blocking(
                RobotCommandBuilder.claw_gripper_open_fraction_command(1.0),
                "開啟夾爪"
            )
            time.sleep(0.7)

            self.send_cmd_blocking(
                RobotCommandBuilder.arm_stow_command(),
                "手臂收納 (Stow)"
            )
            
            self.send_cmd_blocking(
                RobotCommandBuilder.claw_gripper_open_fraction_command(0.0),
                "關閉夾爪"
            )
            time.sleep(0.2)
            
        except Exception as e:
            self.get_logger().error(f"執行回收動作時出錯: {e}")
        finally:
            self.is_fetching = False
            self.current_grasp_target_id = None

    # ---------------------------------------------------------
    # 
    # ---------------------------------------------------------
    def detection_obj_and_img(self, image_sources):
        # 每輪清單初始化
        self.detection_round += 1
        round_id = self.detection_round

        self.detected_objects = []
        obj_index_in_round = 0

        best_obj = None
        best_image_response = None
        best_vision_tform_obj = None
        best_target_id = None
        best_detected = None
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

                suffix = chr(ord('a') + obj_index_in_round)
                obj_id = f"{round_id}{suffix}"
                obj_index_in_round += 1

                detected = {
                    "obj": obj,
                    "id": obj_id,  # 本輪暫時 id
                    "vision_tform_obj": vision_tform_obj,
                    "status": STATUS_DETECTED,
                    "distance": distance,
                }
                self.detected_objects.append(detected)

                if distance < nearest_distance:
                    nearest_distance = distance
                    best_obj = obj
                    best_image_response = resp.image_response
                    best_vision_tform_obj = vision_tform_obj
                    best_detected = detected

        # 掃描完後，更新target_list
        self.merge_detected_targets(
            detected_objects=self.detected_objects,
            new_status=STATUS_DETECTED
        )

        # 再依距離統一更新 target 狀態
        self.update_target_status()

        # 將本輪暫時 id 對應到 global target_list 的 id，只有 pending 才回傳
        if best_detected is not None:
            best_target = self.find_matching_target(best_detected["vision_tform_obj"])
            if best_target is not None and best_target.get("status") == STATUS_PENDING:
                best_target_id = best_target["id"]
                return best_obj, best_image_response, best_vision_tform_obj, best_target_id

        return None, None, None, None

    # ---------------------------------------------------------
    # 更新全局 target_list，5 cm 去重
    # ---------------------------------------------------------
    def merge_detected_targets(self, detected_objects=None, new_status=STATUS_DETECTED):
        if detected_objects is None:
            detected_objects = self.detected_objects

        updated_targets = []

        for detected in detected_objects:
            new_tform = detected["vision_tform_obj"]

            best_match_index = None
            best_match_distance = math.inf

            for i, target in enumerate(self.target_list):
                old_tform = target["vision_tform_obj"]

                dx = new_tform.x - old_tform.x
                dy = new_tform.y - old_tform.y
                dz = new_tform.z - old_tform.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                if distance <= self.duplicate_threshold and distance < best_match_distance:
                    best_match_distance = distance
                    best_match_index = i

            if best_match_index is not None:
                self.target_list[best_match_index]["obj"] = detected["obj"]
                self.target_list[best_match_index]["vision_tform_obj"] = detected["vision_tform_obj"]

                updated_targets.append(self.target_list[best_match_index])

            else:
                target_id = f"target_{self.next_target_id}"
                self.next_target_id += 1

                new_target = {
                    "obj": detected["obj"],
                    "id": target_id,
                    "vision_tform_obj": detected["vision_tform_obj"],
                    "status": new_status,
                    "pose_in_body": None,
                    "last_time": None,
                    "fail_count": 0,
                }
                self.target_list.append(new_target)
                updated_targets.append(new_target)

        return updated_targets
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    def update_target_status(self):
        if len(self.target_list) == 0:
            return

        nearest_target = None
        nearest_distance = math.inf
        now_sec = self.get_clock().now().nanoseconds / 1e9

        for target in self.target_list:
            if target.get("status") in [STATUS_GRASPING, STATUS_GRASPED, STATUS_UNHANDLED]:
                continue

            tform = target["vision_tform_obj"]
            pose_in_body = None
            distance = None

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

                # TF 成功：更新 pose_in_body / last_time
                target["pose_in_body"] = pose_in_body
                target["last_time"] = now_sec

                x = pose_in_body.pose.position.x
                y = pose_in_body.pose.position.y
                distance = math.sqrt(x * x + y * y)

            except Exception as e:
                cached_pose = target.get("pose_in_body", None)
                last_time = target.get("last_time", None)

                # TF 失敗：若舊 pose_in_body 還夠新，就沿用
                if cached_pose is not None and last_time is not None and (now_sec - last_time) < self.tf_cache_timeout_sec:
                    pose_in_body = cached_pose
                    x = pose_in_body.pose.position.x
                    y = pose_in_body.pose.position.y
                    distance = math.sqrt(x * x + y * y)
                else:
                    continue

            if distance <= 1.5:
                target["status"] = STATUS_PENDING
            else:
                target["status"] = STATUS_DETECTED

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_target = target

        if nearest_target is not None and nearest_target.get("status") != STATUS_PENDING:
            nearest_target["status"] = STATUS_APPROACHING

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
    # ---------------------------------------------------------
    #根據id搜索 target_list 中的目標，回傳目標資料
    # ---------------------------------------------------------
    def get_target_by_id(self, target_id):
        for target in self.target_list:
            if target["id"] == target_id:
                return target
        return None
    
    def find_matching_target(self, vision_tform_obj):
        best_target = None
        best_distance = math.inf

        for target in self.target_list:
            old_tform = target["vision_tform_obj"]

            dx = vision_tform_obj.x - old_tform.x
            dy = vision_tform_obj.y - old_tform.y
            dz = vision_tform_obj.z - old_tform.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            if distance <= self.duplicate_threshold and distance < best_distance:
                best_distance = distance
                best_target = target

        return best_target
    
    def handle_grasp_failure(self, target):
        if target is None:
            self.current_grasp_target_id = None
            self.is_fetching = False
            return

        fail_count = target.get("fail_count", 0) + 1
        target["fail_count"] = fail_count

        if fail_count > 5:
            target["status"] = STATUS_UNHANDLED
            self.get_logger().warn(
                f"目標 {target['id']} 夾取失敗超過 5 次，設為 UNHANDLED"
            )
        else:
            target["status"] = STATUS_PENDING
            self.get_logger().warn(
                f"目標 {target['id']} 夾取失敗，第 {fail_count} 次，退回 PENDING"
            )

        self.current_grasp_target_id = None
        self.is_fetching = False

def main(args=None):
    rclpy.init(args=args)
    node = SpotFetchROS2Node()
    
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=8)
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
