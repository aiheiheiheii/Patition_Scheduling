def sort_edges(edges):
    sorted_edges = sorted(edges, key=lambda x: x.end_time)
    return sorted_edges


def find_min_wait_time(edge_end_time, subtasks):
    flag = True  # 标记是不是第一个元素
    min_task = None
    min_wait_time = float('inf')#999999999999999999999999999
    for task in subtasks:
        # 获取任务的前驱任务
        pre_task = task.pre_task
        # 检查前去任务是否存在
        if pre_task is None:
            continue
        # 计算前驱任务的结束时间和边缘节点执行任务的结束时间
        pre_task_end_time = pre_task.end_time
        wait_time = pre_task_end_time - edge_end_time
        if wait_time < 0:
            wait_time = 0
        if flag:
            min_wait_time = wait_time
            min_task = task
            flag = False
        else:
            if wait_time < min_wait_time:
                min_wait_time = wait_time
                min_task = task
    return min_task, min_wait_time


def print_ess(tips, ess):
    print(tips)
    # 更新完成后，可以打印输出每个任务的名称、执行时间和传输时间
    for edge in ess:
        print(f"Edge: {edge.name}")
        tasks = edge.all_tasks
        for task in tasks:
            print(
                f"Task Name: {task.name}, Execution Time: {task.execution_time}, "
                f"Transmission Time: {task.transmission_time}")


def scheduling_subtasks_ess(edges, subtasks):
    # print_ess("查看排序前的服务器上的列表：", edges)
    # 1.对于每个服务器里面的任务进行排序
    for edge in edges:
        tasks = edge.all_tasks  # 服务器上包含的子任务
        # 1.对于每个服务器里面的任务进行分类，分成执行时间 > 服务时间 和服务时间 < 执行时间
        # 第一类 执行时间 > 服务时间 s_1， 第二类 执行时间 < 服务时间 s_2
        s_1 = []
        s_2 = []
        for task in tasks:
            if task.execution_time > task.transmission_time:
                s_1.append(task)
            else:
                s_2.append(task)
        s_1.sort(key=lambda x: x.execution_time, reverse=False)
        s_2.sort(key=lambda x: x.transmission_time, reverse=True)
        # 创建一个新的空列表来存储合并后的结果
        merged_list = []

        # 将 s_1 的元素添加到新列表中
        merged_list.extend(s_1)

        # 将 s_2 的元素添加到新列表中
        merged_list.extend(s_2)

        edge.all_tasks = merged_list

    # print_ess("查看排序后的服务器上的列表：", edges)

    # 2.进行第二个循环，把A类任务全部移到前面
    count = 0  # 记录已排序任务的子任务的数量
    for edge in edges:

        tasks = edge.all_tasks
        # 找出所有classification为'A'的元素,并记录任务执行完毕的时间
        a_list = [task for task in tasks if task.classification == 'A']
        none_a_list = [task for task in tasks if task.classification != 'A']
        merged_list = a_list + none_a_list
        for i, task in enumerate(a_list):
            if i == 0:
                task.start_time = task.transmission_time
            else:
                task.start_time = a_list[i - 1].end_time
            task.end_time = task.start_time + task.execution_time
        edge.all_tasks = merged_list
        edge.end_time = a_list[-1].end_time if a_list else 0
        edge.point = len(a_list)
        count += len(a_list)

        # a_list = []
        # none_a_list = []
        # merge = []
        # flag = True  # 记录节点中的第一个任务
        # for i in range(len(tasks)):
        #     if tasks[i].classification == 'A':
        #         if flag:
        #             tasks[i].start_time = tasks[i].transmission_time
        #             flag = False
        #         else:
        #             tasks[i].start_time = tasks[i - 1].end_time
        #         tasks[i].end_time = tasks[i].start_time + tasks[i].execution_time
        #         a_list.append(tasks[i])
        #     else:
        #         none_a_list.append(tasks[i])
        # merge.extend(a_list)
        # merge.extend(none_a_list)
        # if len(a_list) > 0:
        #     edge.end_time = a_list[-1].end_time
        # edge.point = len(a_list)
        # edge.all_tasks = merge
        # count = count + len(a_list)
    # print_ess("第二次排序查看排序后的服务器上的列表：", edges)
    # 3.进行非A类的子任务进行排序
    while count < len(subtasks):  # 当所有的子任务都排序完成，则循环结束
        after_sort_edges = sort_edges(edges)  # 按照边缘节点当前任务的执行完毕时间进行排序的
        temp_sum = 0
        for edge in after_sort_edges:

            tasks = edge.all_tasks
            # 找出可以排序的子任务，意思是他们的前驱任务已经执行了
            executed_subtask = []
            non_order_subtask = []
            for i in range(edge.point, len(tasks)):  # 在[point,len(tasks)-1]里的元素是没有进行排序的
                non_order_subtask.append(tasks[i])
                # 或者可以改成tasks[i].pre_task.end_time<=edge.end_time,表示前置任务已经执行完毕
                if tasks[i].pre_task.end_time >= 0:  # 表示前驱任务已经执行了
                    executed_subtask.append(tasks[i])
            temp_sum = temp_sum + len(executed_subtask)
            # 如果本次搜索没有可以执行的子任务，则跳出本次循环，执行下一个节点的排序
            if len(executed_subtask) == 0:
                continue

            # 计算等待时间
            # 找出可以排序的子任务后，看看哪个子任务能够使节点的等待时间最小，就排在前面
            min_task, min_wait_time = find_min_wait_time(edge.end_time, executed_subtask)
            min_task.start_time = edge.end_time + min_wait_time
            min_task.end_time = min_task.start_time + min_task.execution_time
            # 把节点的序列按照任务进行排序
            edge.all_tasks[edge.point] = min_task
            edge.point = edge.point + 1
            edge.wait_time = min_wait_time + edge.wait_time
            non_order_subtask.remove(min_task)

            for order_tasks in executed_subtask:
                if min_task.name != order_tasks.name:
                    order_tasks.start_time = edge.all_tasks[edge.point - 1].end_time
                    order_tasks.end_time = order_tasks.start_time + order_tasks.execution_time
                    edge.all_tasks[edge.point] = order_tasks
                    edge.point = edge.point + 1
                    non_order_subtask.remove(order_tasks)

            # 把未排序的任务添加到列表中
            # 获取edge中的指针
            p = edge.point
            for n_task in non_order_subtask:
                edge.all_tasks[p] = n_task
        count = count + temp_sum

    # print_ess("第三次排序查看排序后的服务器上的列表：", edges)

    return edges
