import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import math
import time
import threading

# Boston Dynamics SDK (僅用於視覺與資料處理，不搶 Lease)
import bosdyn.client
import bosdyn.client.util
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2, network_compute_bridge_pb2
from bosdyn.client import frame_helpers
from bosdyn.client.network_compute_bridge_client import (ExternalServerError, NetworkComputeBridgeClient)
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient

# ROS 2 轉換與 Action 訊息
from bosdyn_msgs.conversions import convert
from spot_msgs.action import Manipulation
from spot_msgs.action import RobotCommand

# ros2
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup


# -----------------------  相機來源定義 -------------------------------------------
ImageSources = [
    'frontleft_fisheye_image', 'frontright_fisheye_image'
]
# ----------------------- status 定義 -------------------------------------------
STATUS_DETECTED    = "detected"      # 偵測狀態
STATUS_APPROACHING = "approaching"   # 移動狀態
STATUS_SELECTED    = "selected"      # 選定狀態 
STATUS_GRASPING    = "grasping"      # 夾取狀態
STATUS_POST_GRASP  = "post_grasp"    # 放置狀態
STATUS_GRASPED     = "grasped"       # 完成狀態
STATUS_UNHANDLED   = "unhandled"     # 完成狀態


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
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.ncb_client = self.robot.ensure_client(NetworkComputeBridgeClient.default_service_name)
        
        # --- 3. 任務參數 ---
        self.ml_service     = "fetch-server"       # TODO: 填入你的服務名稱
        self.model_name     = "best.engine"        # TODO: 填入你的模型名稱
        self.target_label   = "Bottle_and_Can"   # TODO: 你要找的標籤

        # --- 4. 啟動主迴圈計時器 --- 
        self.move_msg       = Twist()

        # --- 5. 固定參數 ---
        self.min_confidence         = 0.5
        self.duplicate_threshold    = 0.2
        self.tf_timeout_threshold   = 10.0
        self.distance_threshold     = 1.5
        
        # --- 6. 建立一個並行群組 ---
        self.nav_group      = ReentrantCallbackGroup()
        self.timer_group    = MutuallyExclusiveCallbackGroup()
        
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
        
        # 發布最終給 Spot 的控制速度
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # ----------------------- detection / target list -------------------------------
        self.detected_objects   = []   # 本輪清單
        self.target_list        = []   # 全局清單
        self.next_target_id     = 0
        self.current_target_id  = None

        # ----------------------- TF ---------------------------------------------------
        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(self.tf_buffer, self)
        # ----------------------- Feedback 參數 -----------------------------------------
        self.current_goal_handle = None 
        self._manip_result_future = None
        self.last_feedback_state = None
        self.state_start_time = None
        self.state_timeout_duration = 5.0  # 設定超時秒數
        # -------------------------------------------------------------------------------

    def nav_callback(self, msg):
        """
        核心邏輯：如果沒在夾東西，就把導航的速度直接『搬運』給 Spot。
        如果正在夾東西，則不轉發（或者發布 0,0,0 確保靜止）。
        """
        has_approaching = self.has_target_with_status(STATUS_APPROACHING)
        has_fetching = self.has_target_with_status([ STATUS_SELECTED, STATUS_GRASPING, STATUS_POST_GRASP ])

        if not has_fetching and not has_approaching:
            self.cmd_vel_pub.publish(msg)
        elif has_approaching:
            self.cmd_vel_pub.publish(self.move_msg)
        else:
            # 如果正在處理物品，我們不主動發布 0，
            # 讓 fetch_loop 控制速度就好，避免頻率衝突
            pass
    
    def fetch_loop(self):
        if self.has_target_with_status([STATUS_SELECTED, STATUS_GRASPING, STATUS_POST_GRASP]):
            return
        
        self.get_logger().info('正在透過 NCS 搜尋物體...')

        # 1. 掃描所有看到的目標
        self.detection_obj_and_img(ImageSources)

        # 2. 對每輪新增的 detected_objects 進行去重和合併，更新加入到全局 target_list
        self.merge_detected_targets(
            detected_objects=self.detected_objects,
            new_status=STATUS_DETECTED
        )

        # 3. 更新 target_list 中每個目標的狀態
        self.update_target_status()

        # 4. 若沒有目標，恢復巡邏 (走coverage_path)
        if len(self.target_list) == 0:
                self.get_logger().info("target_list 為空，恢復巡邏模式...")
                return

        # 5. 從 target_list 中找最近目標
        nearest_target = self.find_nearest_target()
        # 5.1 如果最近目標沒有有效的距離或位置資訊，則跳過並繼續巡邏
        if nearest_target is None or nearest_target['pose_in_body'] is None or nearest_target['distance'] is None:
            self.get_logger().info("找不到有效最近目標，恢復巡邏模式...")
            return

        if nearest_target["status"] == STATUS_DETECTED:
            nearest_target["status"] = STATUS_APPROACHING

        # 5.2 印出 target_list 狀態供除錯用
        self.get_logger().info("=== target_list ===")
        for target in self.target_list:
            self.get_logger().info(f"{target['id']}, {target['status']}")
        
        # 6. 如果有有效的最近目標，進入接近模式
        self.get_logger().info(
            f"目前最近目標: {nearest_target['id']}，狀態: {nearest_target['status']}，距離: {nearest_target['distance']:.2f}m"
        )
        # 取得最近目標在 body frame 的位置，計算角度
        tx = nearest_target['pose_in_body'].pose.position.x
        ty = nearest_target['pose_in_body'].pose.position.y
        angle_to_target = math.atan2(ty, tx)
        #-------------------------------------------------------------------------------------------
        #局部規劃移動邏輯：
        # 6.1 如果角度太大 (> 70度)，先原地轉向對齊
        if abs(angle_to_target) > 1.2:
            self.move_msg.linear.x = 0.0
            self.move_msg.angular.z = angle_to_target * 0.6
            return
        # 6.2 若最近目標還大於 1.5m，就接近最近目標
        elif nearest_target['distance'] > self.distance_threshold:
            self.get_logger().info(
                f"目前最近目標: {nearest_target['id']}，"
                f"body x={tx:.2f}, y={ty:.2f}, 距離={nearest_target['distance']:.2f}m"
            )
            # 6.21 介於 70-30度：減速斜向接近
            if abs(angle_to_target) > 0.5:
                self.move_msg.linear.x = 0.2
                self.move_msg.angular.z = angle_to_target * 0.6
            # 6.22 角度小於 30度：直接減速往前接近    
            else:
                self.move_msg.linear.x = 0.3
                self.move_msg.angular.z = angle_to_target * 0.5
            return
        # 6.3 若最近目標小於 0.9m，則太近了，往後退一點並微調角度
        elif nearest_target['distance'] < 0.9:
            self.get_logger().warn(f" 太近了 ({nearest_target['distance']:.2f}m)，往後退一點...")
            self.move_msg.linear.x = -0.2 
            self.move_msg.angular.z = angle_to_target * 0.5
            return  
        
        # 6.4 距離達標 (< 1.5m)，但角度尚未對齊 (> 20度) ---
        elif abs(angle_to_target) > 0.3:
            self.get_logger().info(f"距離達標但偏角 {math.degrees(angle_to_target):.1f}° 太大，原地微調...")
            
            self.move_msg.linear.x = 0.0
            self.move_msg.angular.z = angle_to_target * 0.6 # 慢速微調
            return # 繼續微調，不往下執行夾取
        #-------------------------------------------------------------------------------------------
        # 7. 進入夾取模式
        self.get_logger().info("最近目標已進入 1.5m 範圍，開始進入夾取模式...")
        self.move_msg.linear.x = 0.0
        self.move_msg.angular.z = 0.0
        time.sleep(1.0)
        
        target_obj, image_full, vision_tform_obj, target_id = self.get_obj_and_img(
            ImageSources, nearest_target
        )
        # 7.1 如果無法從 NCS 或 TF 取得有效的目標資訊，則增加失敗計數並嘗試重新接近
        if target_obj is None or vision_tform_obj is None or image_full is None:
            nearest_target['fail_count'] += 1
            self.get_logger().warn(
                f"無法辨識目標 {nearest_target['id']} "
                f"(第 {nearest_target['fail_count']}/5 次嘗試)"
            )

            if nearest_target['fail_count'] >= 5:
                self.get_logger().error(
                    f"目標 {nearest_target['id']} 連續 5 次辨識失敗，標記為 UNHANDLED"
                )
                nearest_target['status'] = STATUS_UNHANDLED
            else:
                nearest_target['status'] = STATUS_DETECTED

            return

        nearest_target['fail_count'] = 0
        nearest_target["status"] = STATUS_SELECTED
        self.current_target_id = target_id
        #------------------------------------------------------------------------------------------
        self.get_logger().info("已抵達目標範圍，開始夾取程序...")

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

    def send_cmd_async(self, sdk_cmd, label):
        """非同步發送：發完指令就直接回傳，不等待結果"""
        goal_msg = RobotCommand.Goal()
        convert(sdk_cmd, goal_msg.command)
        self.get_logger().info(f'發布指令: {label}')
        self.robot_client.send_goal_async(goal_msg) # 不加 callback，不等待
    
    def is_robot_busy(self):
        """檢查目前是否有指令正在 Action Server 執行中"""
        if hasattr(self, 'current_goal_handle') and self.current_goal_handle is not None:
            # 如果 goal_handle 存在，且狀態是在 ACCEPTED 或 EXECUTING (通常 status < 4)
            status = self.current_goal_handle.status
            # GoalStatus.STATUS_ACCEPTED = 1, STATUS_EXECUTING = 2
            return status in [1, 2]
        return False
        
    def send_cmd_blocking(self, sdk_cmd, label):
        """同步阻塞式發送：確保前一個動作完成才回傳"""
        if not self.robot_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('無法連接到 Action Server，請確認 spot_driver 狀態')
            return False    

        goal_msg = RobotCommand.Goal()
        convert(sdk_cmd, goal_msg.command)
        
        if self.is_robot_busy():
            self.get_logger().warn(f'偵測到前一個指令尚未結束，正在等待或取消...')
            # 選項 A: 強制取消 (如果你想插隊)
            # self.current_goal_handle.cancel_goal_async()
            
            # 選項 B: 阻塞直到結束 (如果你想排隊)
            while self.is_robot_busy():
                time.sleep(0.1)
        
        send_goal_future = self.robot_client.send_goal_async(goal_msg)
        while rclpy.ok() and not send_goal_future.done():
            time.sleep(0.1)
            
        self.current_goal_handle = send_goal_future.result()
        if not self.current_goal_handle.accepted:
            self.get_logger().error(f'{label} 被機器人拒絕')
            return False

        result_future = self.current_goal_handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            time.sleep(0.1)
        self.get_logger().info(f'{label} 完成')
        return True
    
    # 建立發送action server
    def send_ros2_manipulation_goal(self, manip_request):
        target = self.get_target_by_id(self.current_target_id)

        if not self.manip_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("找不到 Manipulation Action Server!")
            target['status'] = STATUS_DETECTED
            return
        
        goal_msg = Manipulation.Goal()
        convert(manip_request, goal_msg.command)

        self.get_logger().info('正在發送 ROS 2 夾取指令...')
        send_goal_future = self.manip_client.send_goal_async(
            goal_msg,
            feedback_callback=self.manip_feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def manip_feedback_callback(self, feedback_msg):
        # self.get_logger().info(f"回饋物件內容: {dir(feedback_msg.feedback.feedback)}")
        valid_states = [2, 3, 6, 10, 12]
        state_code = feedback_msg.feedback.feedback.current_state.value
        now = time.time() # 或者使用 self.get_clock().now()
        # self.get_logger().info(f"==> 夾取進度回饋: 狀態碼 {state_code}")
        # --- 狀態卡死檢查邏輯 ---
        if state_code != self.last_feedback_state:
            # 狀態改變了，重置計時器
            self.last_feedback_state = state_code
            self.state_start_time = now
        else:
            # 狀態沒變，檢查是否超時
            if self.state_start_time is not None:
                elapsed = now - self.state_start_time
                if elapsed > self.state_timeout_duration and state_code != 3:
                    self.get_logger().error(f"❌ 夾取狀態卡在 {state_code} 超過 {self.state_timeout_duration} 秒，判定失敗。")
                    self.state_start_time = now
                    self.cancel_current_manipulation()
        
        if state_code not in valid_states:
            self.get_logger().error(f"❌ 偵測到異常狀態碼 {state_code}，判定夾取失敗。")
            self.state_start_time = now
            self.cancel_current_manipulation()

    def cancel_current_manipulation(self):
        """封裝取消動作的輔助函式"""
        if hasattr(self, 'current_goal_handle') and self.current_goal_handle is not None:
            self.current_goal_handle.cancel_goal_async()
            self.get_logger().warn("已發送取消 Action 請求")
            self.current_goal_handle = None # 清除 handle 防止重複呼叫

        self.last_feedback_state = None
        self.state_start_time = None
        
        # 重置目標物狀態，讓它能被重新偵測
        target = self.get_target_by_id(self.current_target_id)
        if target:
            target["status"] = STATUS_DETECTED
            
        action_thread = threading.Thread(target=self.post_grasp_sequence, args=(False,))
        action_thread.start()
        return # 結束此 callback
    
    #檢查 action 發送是否成功
    def goal_response_callback(self, future):
        self.current_goal_handle = future.result()
        if not self.current_goal_handle.accepted:
            self.get_logger().error('夾取請求被拒絕！')

            target = self.get_target_by_id(self.current_target_id)
            if target is not None and target.get("status") not in [STATUS_POST_GRASP, STATUS_GRASPED]:
                target["status"] = STATUS_DETECTED
            self.current_target_id = None
            return

        # action server 已接受，這時才設成 grasping
        target = self.get_target_by_id(self.current_target_id)
        if target is not None:
            target["status"] = STATUS_GRASPING

        self.get_logger().info('夾取請求已接受，執行中...')
        result_future = self.current_goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    # action 結束的檢查       
    def get_result_callback(self, future):
        status = future.result().status
        target = self.get_target_by_id(self.current_target_id)

        if status == 4:
            self.get_logger().info('夾取動作確認成功！')
            time.sleep(0.1)
            robot_state = self.robot_state_client.get_robot_state()
            
            manip_state = robot_state.manipulator_state
            # is_holding = manip_state.is_gripper_holding_item
            open_percent = manip_state.gripper_open_percentage
            if open_percent > 1.5:
                is_holding = True
            else:
                is_holding = False
                
            if not is_holding:
                self.get_logger().warn(f"❌ 動作完成但未偵測到物體 (夾空)，開合度: {open_percent}%")
                # test = input("請按 Enter 繼續...")
                action_thread = threading.Thread(target=self.post_grasp_sequence, args=(is_holding,))
                action_thread.start()
                return
            
            self.get_logger().info(f"✅ 確認夾持成功！開合度: {open_percent}%")
            
            # test = input("請按 Enter 繼續...")
            # 夾取成功後，將目標狀態改成 POST_GRASP 完成夾取準備放置
            if target is not None:
                target["status"] = STATUS_POST_GRASP

            action_thread = threading.Thread(target=self.post_grasp_sequence, args=(is_holding,))
            action_thread.start()

        
    def post_grasp_sequence(self, grasp_success):
        target = self.get_target_by_id(self.current_target_id)
        if grasp_success:
            self.send_cmd_blocking(RobotCommandBuilder.arm_ready_command(), "手臂預備 (Ready)")
            
            try:
                final_joints = [3.103, -1.219, 0.732, 0.013, 1.826, 2.877]
                
                # --- 步驟 1: 高位轉向後方 ---
                # 我們保留 sh0 (3.103)，但將 sh1 設為較高的角度 (例如 -0.5 或 -0.8)
                # 這樣手臂會「舉著」轉到後面，不會掃到背上的設備
                high_rotate_joints = [2.38, -1.719, 0.8, 0.0, 1.826, 2.877] 
                
                self.get_logger().info("回收中")
                cmd1 = RobotCommandBuilder.arm_joint_command(*high_rotate_joints)
                self.send_cmd_async(cmd1, "高位旋轉")
                time.sleep(0.9)
                # --- 步驟 2: 下壓至最終目標點 ---
                # 此時第一軸已經到位，垂直降落即可
                # self.get_logger().info("執行步驟 2: 下壓至背後安全點...")
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
                if target is not None:
                    target["status"] = STATUS_UNHANDLED
                self.current_target_id = None
        else:
            robot_state = self.robot_state_client.get_robot_state()
            manip_state = robot_state.manipulator_state
            stow_status = manip_state.stow_state
            self.send_cmd_blocking(
                RobotCommandBuilder.claw_gripper_open_fraction_command(1.0),
                "開啟夾爪"
            )
            self.send_cmd_blocking(RobotCommandBuilder.arm_ready_command(), "手臂預備 (Ready)")
            # time.sleep(0.5)
            # unstow_joint = [0.0, -2.728982, 2.626391, -0.003963, 0.184514, 0.002903]
            # unstow_cmd = RobotCommandBuilder.arm_joint_command(*unstow_joint, max_vel=20.0, 
            #     max_accel=5.0)
            # self.send_cmd_blocking(unstow_cmd, "unstow")
            # time.sleep(0.5)
            self.send_cmd_blocking(
                RobotCommandBuilder.arm_stow_command(),
                "手臂收納 (Stow)"
            )
            time.sleep(0.3)
            robot_state = self.robot_state_client.get_robot_state()
            manip_state = robot_state.manipulator_state
            stow_status = manip_state.stow_state

            self.send_cmd_blocking(
                    RobotCommandBuilder.claw_gripper_open_fraction_command(0.0),
                    "關閉夾爪"
                )

            target["status"] = STATUS_GRASPED


    # ---------------------------------------------------------
    # 夾取用辨識函式：從 NCS 偵測結果中找到與 nearest_target 最匹配的物體
    # ---------------------------------------------------------
    def get_obj_and_img(self, image_sources, nearest_target):
        best_obj = None
        best_target_id = None
        best_image_response = None
        best_vision_tform_obj = None
        
        min_offset = math.inf # 改為記錄與目標的最小偏移
        

        target_pos = nearest_target['vision_tform_obj']
        
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
            except ExternalServerError as e:
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
                except bosdyn.client.frame_helpers.ValidateFrameTreeError as e:
                    vision_tform_obj = None

                if vision_tform_obj is None:
                    continue

                # --- 核心邏輯修改：對比 NCS 偵測點與當前 Target 點的距離 ---
                ox = vision_tform_obj.x
                oy = vision_tform_obj.y
                oz = vision_tform_obj.z

                # 計算偵測到的物體與 nearest_target 的位移 (Offset)
                offset = math.sqrt(
                    (ox - target_pos.x)**2 + 
                    (oy - target_pos.y)**2 + 
                    (oz - target_pos.z)**2
                )

                if offset > self.duplicate_threshold:
                    continue

                if offset < min_offset:
                    min_offset = offset
                    best_obj = obj
                    best_image_response = resp.image_response
                    best_vision_tform_obj = vision_tform_obj
                    best_target_id = nearest_target["id"]

        if best_obj is None:
            return None, None, None, None

        return best_obj, best_image_response, best_vision_tform_obj, best_target_id
    # --------------------------------------------------------------------------------------------------------
    # detection_obj_and_img:掃描所有影像來源，將符合條件的偵測結果加入 detected_objects 清單
    # --------------------------------------------------------------------------------------------------------
    def detection_obj_and_img(self, image_sources):
        # 重置本輪偵測清單
        self.detected_objects = []

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
            except ExternalServerError as e:
                self.get_logger().error(f'NCS 連線錯誤 ({source}): {e}')
                continue

            if len(resp.object_in_image) > 0:
                for obj in resp.object_in_image:

                    obj_label = obj.name.split('_label_')[-1]   
                    if obj_label != self.target_label:
                        # 不是我們要找的目標，忽略它
                        # 如果有多類別需求，可以在這裡擴充對其他 label 的處理邏輯
                        continue

                    try:
                        vision_tform_obj = frame_helpers.get_a_tform_b(
                            obj.transforms_snapshot,
                            frame_helpers.VISION_FRAME_NAME,
                            obj.image_properties.frame_name_image_coordinates
                        )
                    except bosdyn.client.frame_helpers.ValidateFrameTreeError as e:
                        vision_tform_obj = None

                    if vision_tform_obj is None:
                        continue

                    self.detected_objects.append({
                        "obj": obj,
                        "vision_tform_obj": vision_tform_obj,
                        "status": STATUS_DETECTED,
                    })
    # --------------------------------------------------------------------------------------------------------
    # merge_detected_targets:對每輪新增的 detected_objects 進行去重和合併，更新全局 target_list
    # --------------------------------------------------------------------------------------------------------
    def merge_detected_targets(self, detected_objects=None, new_status=STATUS_DETECTED):

        if detected_objects is None:
            detected_objects = self.detected_objects

        updated_targets = []

        for detected in detected_objects:
            new_tform = detected["vision_tform_obj"]

            best_match_index = None
            best_match_distance = math.inf

            for i, target in enumerate(self.target_list):
                # 設定不更新的狀態集合{grasped, unhandled}
                if target.get("status") in [STATUS_GRASPED, STATUS_UNHANDLED]:
                    continue
                # 取得 target_list中每個目標的vision_tform_obj
                old_tform = target["vision_tform_obj"]
                # 計算新舊目標的距離(3D距離)
                dx = new_tform.x - old_tform.x
                dy = new_tform.y - old_tform.y
                dz = new_tform.z - old_tform.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                # 如果小於重複門檻(duplicate_threshold)和目前最佳距離(best_match_distance)，則更新數值
                if distance <= self.duplicate_threshold and distance < best_match_distance:
                    best_match_distance = distance
                    best_match_index = i
            # 匹配到的話就更新目標資料(obj, vision_tform_obj)不調整status
            if best_match_index is not None:
                self.target_list[best_match_index]["obj"] = detected["obj"]
                self.target_list[best_match_index]["vision_tform_obj"] = detected["vision_tform_obj"]

                updated_targets.append(self.target_list[best_match_index])
            # 無匹配到的話就新增一筆目標資料，並給予新的ID和狀態(detected)
            else:
                target_id = f"target_{self.next_target_id}"
                self.next_target_id += 1

                new_target = {
                    "obj": detected["obj"],
                    "id": target_id,
                    "vision_tform_obj": detected["vision_tform_obj"],
                    "status": new_status,
                    "last_time": None,
                    "pose_in_body": None,
                    "distance": None,
                    "fail_count": 0,
                    "failure_reason": None,
                }
                self.target_list.append(new_target)
                updated_targets.append(new_target)

        return updated_targets
    # --------------------------------------------------------------------------------------------------------
    # update_target_status:根據距離更新 target_list 中每個目標的狀態
    # --------------------------------------------------------------------------------------------------------
    def update_target_status(self):
        # target_list 中如果沒有任何目標，就直接回傳
        if len(self.target_list) == 0:
            return
        # 取得 ROS 2 clock 的現在時間，並轉成秒
        now_sec = self.get_clock().now().nanoseconds / 1e9

        for target in self.target_list:
            # 設定不更新狀態的狀態集合
            if target.get("status") in [
                STATUS_SELECTED, STATUS_GRASPING, STATUS_POST_GRASP, STATUS_GRASPED, STATUS_UNHANDLED
                ]:
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
                target["distance"] = distance

            except Exception :
                cached_pose = target.get("pose_in_body", None)
                last_time = target.get("last_time", None)
                #TF 失敗：若舊 pose_in_body 還夠新，就沿用
                if cached_pose is not None and last_time is not None and (now_sec - last_time) < self.tf_timeout_threshold:
                    pose_in_body = cached_pose
                    x = pose_in_body.pose.position.x
                    y = pose_in_body.pose.position.y
                    distance = math.sqrt(x * x + y * y)
                    target["distance"] = distance
                else:
                    continue        


            target["status"] = STATUS_DETECTED
    # --------------------------------------------------------------------------------------------------------
    # find_nearest_target:搜尋全局 target_list 中最近的目標
    # --------------------------------------------------------------------------------------------------------
    def find_nearest_target(self):
        if not self.target_list:
            return None

        candidates = self.get_targets_by_status([
            STATUS_APPROACHING,
            STATUS_DETECTED
        ])

        nearest_target = None
        nearest_distance = math.inf

        for target in candidates:
            distance = target.get("distance")
            pose_in_body = target.get("pose_in_body")

            if pose_in_body is None or distance is None:
                continue

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_target = target

        return nearest_target
    # ---------------------------------------------------------
    # find_center_px:計算 NCS 偵測到的物體在影像中的像素中心點
    # ---------------------------------------------------------
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
    # target_list 的管理函數
    # ---------------------------------------------------------
    def get_target_by_id(self, target_id):
        for target in self.target_list:
            if target["id"] == target_id:
                return target
        return None

    def get_targets_by_status(self, statuses):
        if isinstance(statuses, str):
            statuses = [statuses]
        return [t for t in self.target_list if t.get("status") in statuses]

    def has_target_with_status(self, statuses: list[str]):
        return any(t.get("status") in statuses for t in self.target_list)
    # ---------------------------------------------------------        

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
