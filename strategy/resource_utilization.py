"""
    配置服务器的信息：包括ip、用户名和密码
    获取服务器资源使用情况
"""
import paramiko
from config import server_list

# 获取服务器资源使用情况的命令
commands = {
    # 获取 CPU 使用率
    "cpu": "top -bn1 | grep load | awk '{printf \"CPU Load: %.2f\", $(NF-2)}'",
    # 获取 GPU 使用率(需要服务器上安装 nvidia-smi 命令)
    "gpu": "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
    # 获取内存使用率
    "memory": "free -m | awk 'NR==2{printf \"Memory Usage: %s/%sMB (%.2f%%)\", $3,$2,$3*100/$2 }'",
    # # 获取网络信息,服务器需要安装 ifstat
    "network": "ifstat 1 1"
}


# 获取服务器的资源利用率
def get_single_server_info(server_info):
    # 创建一个SSH客户端对象
    client = paramiko.SSHClient()
    # 设置自动添加主机名和主机秘钥到本地HostKeys对象,并保存
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        # 连接SSH客户端
        client.connect(server_info["ip"], username=server_info["username"], password=server_info["password"])
        server_resources = {"ip": server_info["ip"]}

        for resource, command in commands.items():
            # 执行命令并获取标准输入、标准输出和标准错误
            stdin, stdout, stderr = client.exec_command(command)
            # 读取标准输出并解码为字符串,去除首尾空白字符
            output = stdout.read().decode().strip()
            # 读取标准错误并解码为字符串,去除首尾空白字符
            error = stderr.read().decode().strip()
            if error:
                server_resources[resource] = f"Error: {error}"
            else:
                server_resources[resource] = output

        return server_resources
    except Exception as e:
        print(f"Couldn't connect to {server_info['ip']}: {e}")
        return None
    finally:
        client.close()


# 获取全部服务器的资源利用率
def get_all_server_info():
    # 创建一个字典来存储所有服务器的信息
    infos = {}
    # 遍历服务器列表,对每个服务器执行信息获取
    for server_info in server_list:
        # 调用 get_info 函数获取单个服务器的信息
        server_resources = get_single_server_info(server_info)

        if server_resources:
            # 如果成功获取到服务器信息,将结果存入 infos 字典
            infos[server_info["ip"]] = {
                "cpu": server_resources.get("cpu", ""),
                "gpu": server_resources.get("gpu", ""),
                "memory": server_resources.get("memory", ""),
                "network": server_resources.get("network", "")
            }

    return infos
