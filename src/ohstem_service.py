import paho.mqtt.client as mqtt

class OhstemService:
    def __init__(self, username: str, password: str = "", feed_id: str = "V1"):
        self.username = username
        self.password = password
        self.feed_id = feed_id
        self.broker = "mqtt.ohstem.vn"
        self.port = 1883
        self.client = mqtt.Client()
        
        if self.password:
            self.client.username_pw_set(self.username, self.password)
            
    def connect(self) -> bool:
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            print(f"[Ohstem] Kết nối thành công tới MQTT Broker: {self.broker}")
            return True
        except Exception as e:
            print(f"[Ohstem] Kết nối tới MQTT Broker thất bại: {e}")
            return False

    def send_attendance_count(self, count: int):
        """
        Gửi sĩ số học sinh lên Ohstem Dashboard qua feed_id cấu hình
        """
        if not self.username:
            return
        topic = f"{self.username}/feeds/{self.feed_id}"
        try:
            self.client.publish(topic, str(count))
            print(f"[Ohstem] Đã gửi sĩ số: {count} tới Feed {self.feed_id}")
        except Exception as e:
            print(f"[Ohstem] Lỗi gửi dữ liệu lên OhStem: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("[Ohstem] Đã ngắt kết nối MQTT.")
