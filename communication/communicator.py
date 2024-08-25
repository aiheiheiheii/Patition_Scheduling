# Correspondence Object
import pickle
import time
import struct
import socket
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Communicator(object):
	def __init__(self, host_ip, host_port):
		# 创建socket对象
		self.sock = socket.socket()
		# 设置socket以便地址重用
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		# 绑定socket到特定IP和端口
		try:
			self.sock.bind((host_ip, host_port))
		except socket.error as e:
			logger.error(f"Socket error: {e}")
			raise
		logger.info(f"host_ip:{host_ip}, host_port:{host_port}")

	def wait_for_connection(self):
		"""
		监听并接受客户端连接
		"""
		self.sock.listen()
		logger.info('Waiting for incoming connection...')
		try:
			node_sock, node_address = self.sock.accept()
		except Exception as e:
			logger.error(f'Error accepting connection: {e}')
			return None, None
		logger.info(f'Connection from {node_address}')
		return node_sock, node_address

	def send_message(self, sock, msg):
		"""
		发送消息
		"""
		try:
			# 将消息序列化
			msg_pickle = pickle.dumps(msg)
			# 首先发送字符串的长度
			sock.sendall(struct.pack(">I", len(msg_pickle)))
			# 然后发送字符串本身
			sock.sendall(msg_pickle)
			# 记录发送日志
			logger.debug(f'{msg[0]} sent to {sock.getpeername()[0]}:{sock.getpeername()[1]}')
		except socket.error as e:
			logger.error(f"Error sending message: {e}")

	def receive_message(self, sock, expect_msg_type=None):
		"""
		接收数据
		:param sock:
		:param expect_msg_type:
		:return:
		"""
		try:
			# 接收消息长度信息
			msg_len_bytes = sock.recv(4)
			if len(msg_len_bytes) < 4:
				# 如果接收到的长度信息不完整,表示连接已关闭
				return None
			msg_len = struct.unpack(">I", msg_len_bytes)[0]

			# 接收完整消息
			msg_bytes = bytearray()
			while len(msg_bytes) < msg_len:
				chunk = sock.recv(msg_len - len(msg_bytes))
				if not chunk:
					# 如果接收到的数据为空,表示连接已关闭
					return None
				msg_bytes.extend(chunk)

			# 反序列化消息
			msg = pickle.loads(msg_bytes)
			# 记录接收日志
			logger.debug(msg[0] + 'received from' + str(sock.getpeername()[0]) + ':' + str(sock.getpeername()[1]))
			return msg
		except (socket.error, struct.error) as e:
			# 处理连接关闭或数据不完整的情况
			logger.error(f"Error receiving message: {e}")
			return None


class NodeEnd(Communicator):
	def __init__(self, ip, port):
		# 调用父类communicator的构造函数来初始化
		super(NodeEnd, self).__init__(ip, port)

	def node_connect(self, node_ip, node_port, max_retries=10):
		# 尝试连接的次数
		attempts = 0
		while attempts < max_retries:
			try:
				# 尝试创建到下一个节点的连接
				self.sock.connect((node_ip, node_port), )
				# 如果连接成功，则跳出循环
				print(f"已成功连接到{node_ip}:{node_port}")
				# If the connection is successful, break the loop
				break
			except socket.error as e:
				# 如果连接失败，打印错误消息并重试
				print(f"连接到{node_ip}:{node_port}失败，正在重试...错误：{e}")
				# 等待一段时间再次尝试
				time.sleep(1)
				# 增加尝试连接的次数
				attempts += 1
		if attempts == max_retries:
			# 如果达到最大重试次数，则抛出异常
			raise Exception(f"无法连接到{node_ip}:{node_port}， 已达到最大重试次数{max_retries}")