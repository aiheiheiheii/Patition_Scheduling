import pickle
import struct
import logging

# 设置日志记录器
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def send_data(sock, data):
    """
    发送数据到指定的socket
    """
    try:
        # 将数据序列化
        serialized_data = pickle.dumps(data)
        # 首先发送数据的长度
        sock.sendall(struct.pack(">I", len(serialized_data)))
        # 然后发送数据本身
        sock.sendall(serialized_data)
    except Exception as e:
        logger.error(f"Error sending data: {e}")


def receive_data(sock):
    """
    从指定的socket接收数据
    """
    try:
        # 接收数据长度信息
        data_len_bytes = sock.recv(4)
        if len(data_len_bytes) < 4:
            # 数据长度信息不完整，可能连接已关闭
            return None
        data_len = struct.unpack(">I", data_len_bytes)[0]

        # 接收完整数据
        data_bytes = bytearray()
        while len(data_bytes) < data_len:
            chunk = sock.recv(data_len - len(data_bytes))
            if not chunk:
                # 如果接收到的数据为空，可能连接已关闭
                return None
            data_bytes.extend(chunk)

        # 反序列化数据
        data = pickle.loads(data_bytes)
        return data
    except Exception as e:
        logger.error(f"Error receiving data: {e}")
        return None
