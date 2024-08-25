import paramiko
import psutil
import time
from config import server_list


# LINUX 命令
def execute_command(ssh, command):
    """在远程服务器上执行命令并返回结果"""
    stdin, stdout, stderr = ssh.exec_command(command)
    return stdout.read().decode().strip()


def get_cpu_usage(ssh):
    """获取远程服务器的CPU使用率"""
    command = "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1 \"%\"}'"
    return execute_command(ssh, command)


def get_memory_usage(ssh):
    """获取远程服务器的内存使用情况"""
    command = "free -m | awk 'NR==2{printf \"Total: %sMB, Used: %sMB, Free: %sMB, Usage: %.2f%%\", $2, $3, $4, $3*100/$2 }'"
    return execute_command(ssh, command)


def get_network_interface(ssh):
    """获取远程服务器的网络接口名称"""
    command = "ls /sys/class/net | grep -E '^(eth|en|wlan)' | head -n 1"
    return execute_command(ssh, command)


def format_bandwidth(bytes_per_sec):
    """格式化带宽值并添加适当的单位"""
    units = ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"]
    scale = 1024.0
    for unit in units:
        if bytes_per_sec < scale:
            return f"{bytes_per_sec:.2f} {unit}"
        bytes_per_sec /= scale


def get_network_bandwidth(ssh, interface, interval=1):
    """获取远程服务器的网络带宽信息"""
    command = f"cat /proc/net/dev | grep {interface} | awk '{{print $2 \" \" $10}}'"
    net_io_before = execute_command(ssh, command).split()
    time.sleep(interval)
    net_io_after = execute_command(ssh, command).split()

    if len(net_io_before) < 2 or len(net_io_after) < 2:
        raise ValueError("网络接口数据读取失败")

    bytes_sent_per_sec = (int(net_io_after[1]) - int(net_io_before[1])) / interval
    bytes_recv_per_sec = (int(net_io_after[0]) - int(net_io_before[0])) / interval

    return {
        'bytes_sent_per_sec': format_bandwidth(bytes_sent_per_sec),
        'bytes_recv_per_sec': format_bandwidth(bytes_recv_per_sec)
    }


def parse_memory_usage(memory_usage_str):
    """解析内存使用情况字符串为字典"""
    parts = memory_usage_str.split(", ")
    memory_info = {}
    for part in parts:
        key, value = part.split(": ")
        memory_info[key] = value
    return memory_info


def get_server_info_by_ip(ip):
    """
    根据给定的 IP 地址从服务器列表中获取相应的服务器信息。

    :param ip: 要查找的服务器 IP 地址
    :return: 对应的服务器信息字典，如果未找到则返回 None
    """
    for server in server_list:
        if server['ip'] == ip:
            return server
    return None


# 获取服务器的资源利用率
def get_linux_resource_info(server_ip):
    # 创建一个SSH客户端对象
    client = paramiko.SSHClient()
    # 设置自动添加主机名和主机秘钥到本地HostKeys对象,并保存
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    server_info = get_server_info_by_ip(server_ip)
    if not server_info:
        print(f"Server info for IP {server_ip} not found.")
        return None
    try:
        # 连接SSH客户端
        client.connect(server_ip, username=server_info["username"], password=server_info["password"])
        server_resources = {}

        # 获取 CPU 使用率
        server_resources['cpu'] = get_cpu_usage(ssh=client)

        # 获取内存使用情况
        memory_usage_str = get_memory_usage(ssh=client)
        server_resources['memory'] = parse_memory_usage(memory_usage_str)

        # 获取网络接口名称
        interface = get_network_interface(ssh=client)
        if not interface:
            print("未找到有效的网络接口")
            return server_resources

        # 获取网络带宽
        network_bandwidth = get_network_bandwidth(ssh=client, interface=interface)
        server_resources['network'] = network_bandwidth

        return server_resources
    except Exception as e:
        print(f"Couldn't connect to {server_info['ip']}: {e}")
        return None
    finally:
        client.close()


def get_system_usage():
    # 获取CPU使用情况
    cpu_usage = psutil.cpu_percent(interval=1)

    # 获取内存使用情况
    memory_info = psutil.virtual_memory()
    memory_total = memory_info.total / 1024 / 1024  # 转换为MB
    memory_used = memory_info.used / 1024 / 1024  # 转换为MB
    memory_free = memory_info.available / 1024 / 1024  # 转换为MB
    memory_usage = memory_info.percent

    return cpu_usage, memory_total, memory_used, memory_free, memory_usage


def get_network_usage(interval=1):
    net_io_before = psutil.net_io_counters()
    time.sleep(interval)
    net_io_after = psutil.net_io_counters()

    bytes_sent = net_io_after.bytes_sent - net_io_before.bytes_sent
    bytes_recv = net_io_after.bytes_recv - net_io_before.bytes_recv

    return bytes_sent, bytes_recv


def get_windows_resource_info():
    cpu_usage, memory_total, memory_used, memory_free, memory_usage = get_system_usage()
    bytes_sent, bytes_recv = get_network_usage()

    formatted_sent = format_bandwidth(bytes_sent)
    formatted_recv = format_bandwidth(bytes_recv)

    result = {
        "cpu": f"{cpu_usage}%",
        "memory": {
            "Total": f"{memory_total:.2f}MB",
            "Used": f"{memory_used:.2f}MB",
            "Free": f"{memory_free:.2f}MB",
            "Usage": f"{memory_usage:.2f}%"
        },
        "network": {
            "bytes_sent_per_sec": formatted_sent,
            "bytes_recv_per_sec": formatted_recv
        }
    }

    return result