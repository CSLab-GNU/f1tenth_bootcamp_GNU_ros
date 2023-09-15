import numpy as np
import math
# from main import GymRunner


class FGM_REF_WAY:
    def __init__(self, params=None):
        # 기본 상수값 및 파라미터 초기화
        self.RACECAR_LENGTH = 0.3302  # 자동차 길이
        self.PI = 3.141592  # 원주율
        self.ROBOT_SCALE = 0.2032  # 로봇 크기
        
        self.LOOK = 0.5  # 참조로 하는 목표 웨이포인트를 현재 차 기준으로 얼마나 멀리 볼것인지에 대한 LOOKAHEAD값 
        self.THRESHOLD = 4.5  # 갭을 판단하기 위해 사용하는 장애물과의 거리를 나타내는 임계값 (THRESHOLD보다 작은 값의 SCAN값은 갭이 될 수 없음)
        self.gap_threshold_size = 30 # 갭을 판단하기 위해 사용되는 연속되는 LiDAR index의 최소값 (크기가 30 이상이어야 GAP으로 인정함)



        self.FILTER_SCALE = 1.1  
        
        # 각종 가중치 및 파라미터
        self.GAP_THETA_GAIN = 20.0
        self.REF_THETA_GAIN = 1.5
        self.BEST_POINT_CONV_SIZE = 160

        # 웨이포인트 경로
        self.waypoint_real_path = 'pkg/Oschersleben_2_wp.csv'
        self.waypoint_delimeter = ','

        # 초기값 설정
        self.scan_range = 1080
        self.desired_gap = 0
        self.desired_wp_rt = [0, 0]
        self.wp_num = 1
        self.wp_index_current = 0
        self.current_position = [0] * 3
        self.nearest_distance = 0
        self.max_angle = 0
        self.wp_angle = 0
        self.gaps = []
        self.alt_gap = [0, 0, 0]
        self.interval = 0.00435  
        self.front_idx = 0
        self.theta_for = self.PI / 3
        self.current_speed = 0
        self.dmin_past = 0
        
        # 웨이포인트 로드
        self.waypoints = self.get_waypoint()


    def get_waypoint(self): 
        # 웨이포인트 CSV 파일에서 웨이포인트 정보 로드
        file_wps = np.genfromtxt(self.waypoint_real_path, delimiter=self.waypoint_delimeter, dtype='float')
        temp_waypoint = []
        for i in file_wps:
            wps_point = [i[0], i[1], 0]
            temp_waypoint.append(wps_point)
            self.wp_num += 1
        return temp_waypoint

    def getDistance(self, a, b): 
        # 두 점 사이의 거리 계산
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return np.sqrt(dx ** 2 + dy ** 2)

    def transformPoint(self, origin, target):
        # 좌표 변환 함수
        theta = self.PI / 2 - origin[2]
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        dtheta = target[2] + theta

        tf_point_x = dx * np.cos(theta) - dy * np.sin(theta)
        tf_point_y = dx * np.sin(theta) + dy * np.cos(theta)
        tf_point_theta = dtheta
        tf_point = [tf_point_x, tf_point_y, tf_point_theta]

        return tf_point

    def xyt2rt(self, origin):
        # XY 좌표에서 극좌표로 변환
        rtpoint = []
        x = origin[0]
        y = origin[1]
        rtpoint.append(np.sqrt(x * x + y * y))
        rtpoint.append(np.arctan2(y, x) - (self.PI / 2))
        return rtpoint

    def find_desired_wp(self):
        """
        현재 목표로 하는 waypoint를 찾는 함수
        """
        wp_index_temp = self.wp_index_current
        self.nearest_distance = self.getDistance(self.waypoints[wp_index_temp], self.current_position) 
        
        # 가장 가까운 웨이포인트를 찾는 루프
        while True:
            wp_index_temp += 1
            if wp_index_temp >= self.wp_num - 1:
                wp_index_temp = 0
            temp_distance = self.getDistance(self.waypoints[wp_index_temp], self.current_position) 
            if temp_distance < self.nearest_distance:
                self.nearest_distance = temp_distance
                self.wp_index_current = wp_index_temp
            elif (temp_distance > (self.nearest_distance + self.LOOK * 1.2)) or (wp_index_temp == self.wp_index_current):
                break

        # 탐색 거리 내의 웨이포인트 찾기
        idx_temp = self.wp_index_current
        while True:
            if idx_temp >= self.wp_num - 1:
                idx_temp = 0
            temp_distance = self.getDistance(self.waypoints[idx_temp], self.current_position)
            if temp_distance > self.LOOK: break
            idx_temp += 1
    
        transformed_nearest_point = self.transformPoint(self.current_position, self.waypoints[idx_temp])
        self.desired_wp_rt = self.xyt2rt(transformed_nearest_point)


    def subCallback_scan(self, scan_data):
        """
        현재 스캔 데이터와 주변 스캔 데이터를 비교하면서 노이즈가 있는 경우, 해당 스캔 데이터를 인근 값으로 대체하여 LiDAR 데이터 전처리 
        """
        self.front_idx = (int(self.scan_range / 2))
        self.scan_origin = [0] * self.scan_range
        self.scan_filtered = [0] * self.scan_range
        for i in range(self.scan_range):
            self.scan_origin[i] = scan_data[i]
            self.scan_filtered[i] = scan_data[i]
        for i in range(self.scan_range - 1):
            # 노이즈 필터링
            if self.scan_origin[i] * self.FILTER_SCALE < self.scan_filtered[i + 1]:
                unit_length = self.scan_origin[i] * self.interval
                filter_num = self.ROBOT_SCALE / unit_length
                j = 1
                while j < filter_num + 1:
                    if i + j < self.scan_range:
                        if self.scan_filtered[i + j] > self.scan_origin[i]:
                            self.scan_filtered[i + j] = self.scan_origin[i]
                        else:
                            break
                    else:
                        break
                    j += 1
            elif self.scan_filtered[i] > self.scan_origin[i + 1] * self.FILTER_SCALE:
                unit_length = self.scan_origin[i + 1] * self.interval
                filter_num = self.ROBOT_SCALE / unit_length
                j = 0
                while j < filter_num + 1:
                    if i - j > 0:
                        if self.scan_filtered[i - j] > self.scan_origin[i + 1]:
                            self.scan_filtered[i - j] = self.scan_origin[i + 1]
                        else:
                            break
                    else:
                        break
                    j += 1
        return self.scan_filtered

    def find_gap(self, scan):
        """
        조건에 만족하는 gaps를 찾는 함수
        """
        self.gaps = []
        i = 0
        while i < self.scan_range:
            # 장애물과의 거리가 임계값보다 크면 갭으로 판단
            if scan[i] > self.THRESHOLD:
                start_idx_temp = i
                end_idx_temp = i
                while ((scan[i] > self.THRESHOLD) and (i + 1 < self.scan_range)):
                    i += 1
                if scan[i] > self.THRESHOLD:
                    i += 1
                end_idx_temp = i
                gap_size = np.fabs(end_idx_temp - start_idx_temp)
                if gap_size < self.gap_threshold_size:  # 작은 갭은 무시
                    i += 1
                    continue
                gap_temp = [0] * 2
                gap_temp[0] = start_idx_temp
                gap_temp[1] = end_idx_temp
                self.gaps.append(gap_temp)
            i += 1

    def find_gap_failure(self):
        """
        gap을 찾지 못했을 경우 대안을 찾는 함수
        """
        start_idx_temp = (self.front_idx) - 240
        end_idx_temp = (self.front_idx) + 240
        self.alt_gap[0] = start_idx_temp
        self.alt_gap[1] = end_idx_temp

    

    def find_best_gap(self, ref): 
        """
        주어진 참조 방향(ref)에 따라 가장 적절한 갭을 찾는 함수
        """
        num = len(self.gaps)

        if num == 0:  # 갭을 찾지 못했을 경우 대체값 반환
            return self.alt_gap
        else:
            step = (int(ref[1] / self.interval))
            ref_idx = self.front_idx + step
            gap_idx = 0

            # 가장 참조 방향에 근접한 갭을 찾는 과정
            if self.gaps[0][0] > ref_idx:
                distance = self.gaps[0][0] - ref_idx
            elif self.gaps[0][1] < ref_idx:
                distance = ref_idx - self.gaps[0][1]
            else:
                distance = 0
                gap_idx = 0

            i = 1
            while (i < num):  # distance를 갱신하면서 가장 가까운 갭 선택
                if self.gaps[i][0] > ref_idx:
                    temp_distance = self.gaps[i][0] - ref_idx
                    if temp_distance < distance:
                        distance = temp_distance
                        gap_idx = i
                elif self.gaps[i][1] < ref_idx:
                    temp_distance = ref_idx - self.gaps[i][1]
                    if temp_distance < distance:
                        distance = temp_distance
                        gap_idx = i
                else:
                    temp_distance = 0
                    distance = 0
                    gap_idx = i
                    break
                i += 1
            return self.gaps[gap_idx]

    def find_best_point(self, best_gap): 
        """
        주어진 갭 내에서 최적의 포인트(인덱스)를 찾는 함수 
        """
        averaged_max_gap = np.convolve(self.scan_filtered[best_gap[0]:best_gap[1]], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE
        
        return averaged_max_gap.argmax() + best_gap[0]

    def calculate_steering_and_speed(self, best_point):
        """
        최적의 포인트를 기반으로 스티어링 각도와 속도를 계산하는 함수
        """
        self.gap_angle = (best_point - self.front_idx) * self.interval  
        self.wp_angle = self.desired_wp_rt[1]

        # 장애물과의 최소 거리 계산
        dmin = 0
        for i in range(10):
            dmin += self.scan_filtered[i]
        dmin /= 10

        i = 0
        temp_avg = 0
        while i < self.scan_range - 7:
            j = 0
            while j < 10:
                if i + j > 1079:
                    temp_avg += 0
                else:
                    temp_avg += self.scan_filtered[i + j]
                j += 1

            temp_avg /= 10  # 인덱스 i로부터 10개 인덱스의 거리 데이터 평균

            if dmin > temp_avg:
                if temp_avg == 0:
                    temp_avg = dmin
                dmin = temp_avg
            temp_avg = 0
            i += 3

        # 조종 각도 계산
        controlled_angle = ((self.GAP_THETA_GAIN / dmin) * self.gap_angle + self.REF_THETA_GAIN * self.wp_angle) / (
                self.GAP_THETA_GAIN / dmin + self.REF_THETA_GAIN)
 
        # 예상 이동 거리 계산. 기본 거리에 현재 속도를 고려한 값을 더함.
        distance = 1.0 + (self.current_speed * 0.001)

        # 조향 각도를 기반으로 차량이 따라갈 경로의 반경 계산.
        path_radius = distance / (2 * np.sin(controlled_angle))

        # 차량의 길이와 경로의 반경을 사용하여 실제 조향각 계산.
        steering_angle = np.arctan(self.RACECAR_LENGTH / path_radius)

        # 최종적으로 계산된 조향각을 저장.
        steer = steering_angle
        
        self.dmin_past = dmin
        speed = 8
    
        return steer, speed 


    def main_drive(self, obs):
        """
        주요 드라이브 로직. LiDAR 스캔 데이터와 오도메트리 데이터를 입력받아, 
        적절한 속도와 조향각을 결정하는 함수.

        :param scan_data: LiDAR 스캔 데이터
        :param odom_data: 오도메트리(차량 위치 및 방향) 데이터
        :return: 계산된 속도와 조향각
        """

        ranges = obs['scans']

        # LiDAR 데이터 전처리
        scan_data = self.subCallback_scan(ranges)
        
        # 현재 차량의 위치와 방향 정보 업데이트
        self.current_position = [obs['poses_x'], obs['poses_y'], obs['poses_theta']]
        
        # 현재 차량의 속도 정보 업데이트
        self.current_speed = obs['linear_vels_x']

        # 현재 차량 위치에 가장 가까운 웨이포인트 및 목표 웨이포인트 계산
        self.find_desired_wp()
        
        # LiDAR 데이터를 사용하여 주변의 가능한 갭을 탐지
        self.find_gap(ranges)
        
        # 갭을 찾지 못한 경우 대체 방안 적용
        self.find_gap_failure()

        # 찾아낸 갭 중에서 최적의 갭(목표 웨이포인트 방향에 가장 근접한)을 선택
        self.desired_gap = self.find_best_gap(self.desired_wp_rt)
        
        # 선택된 갭 내에서 최적의 포인트를 결정
        self.best_point = self.find_best_point(self.desired_gap)

        # 최적의 포인트를 기반으로 조향각과 속도 계산
        steer, speed = self.calculate_steering_and_speed(self.best_point)

        return speed, steer