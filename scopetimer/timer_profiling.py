import json
import re
import argparse
from datetime import datetime


def parse_glog_timestamp(glog_timestamp):
    """Convert the new log timestamp to Unix epoch in microseconds."""
    dt = datetime.strptime(glog_timestamp, '%Y-%m-%d %H:%M:%S,%f')
    return int(dt.timestamp() * 1e6)


def process_log_file(timer_log_file, resource_log_file, output_file):
    events = []
    thread_id_map = {}
    next_tid = 0

    with open(timer_log_file, "r") as file:
        for line in file:
            # Match for timer logs
            timer_match = re.match(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[PID: (\d+)\] - <<Timer>> (.+): (start|(\d+\.\d+)s), id-(\d+)",
                line)

            if timer_match:
                glog_timestamp, pid, full_name, event_type, duration, id = timer_match.groups(
                )
                timestamp = parse_glog_timestamp(glog_timestamp)
                id = int(id)
                thread_id = int(pid)

                # Extract the leaf name from the full name
                leaf_name = full_name.split('.')[-1]

                # Assign a sequential thread ID if not already assigned
                if thread_id not in thread_id_map:
                    thread_id_map[thread_id] = next_tid
                    next_tid += 1

                sequential_tid = thread_id_map[thread_id]

                if event_type == "start":
                    events.append({
                        "name": leaf_name,
                        "cat": "function",
                        "ph": "B",
                        "ts": timestamp,
                        "pid": 0,
                        "tid": sequential_tid,
                        "args": {
                            "id": id,
                            "full_name": full_name
                        }
                    })
                else:
                    events.append({
                        "name": leaf_name,
                        "cat": "function",
                        "ph": "E",
                        "ts": timestamp,
                        "pid": 0,
                        "tid": sequential_tid,
                        "args": {
                            "id": id,
                            "full_name": full_name
                        }
                    })

    with open(resource_log_file, "r") as file:
        for line in file:
            # Match for timer logs
            # Match for resource logs
            resource_match = re.match(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[PID: (\d+)\] - <<Resource>> Mem: (\d+(\.\d+)?) GB, CPU: (\d+(\.\d+)?) %",
                line)

            if resource_match:
                glog_timestamp, pid, memory_usage, _, cpu_usage, _ = resource_match.groups(
                )
                timestamp = parse_glog_timestamp(glog_timestamp)
                thread_id = int(pid)

                # Assign a sequential thread ID if not already assigned
                if thread_id not in thread_id_map:
                    thread_id_map[thread_id] = next_tid
                    next_tid += 1

                sequential_tid = thread_id_map[thread_id]

                # Memory usage counter event
                events.append({
                    "name": "Memory Usage",
                    "cat": "resource",
                    "ph": "C",
                    "ts": timestamp,
                    "pid": 0,
                    "tid": sequential_tid,
                    "args": {
                        "GB": float(memory_usage)
                    }
                })

                # CPU usage counter event
                events.append({
                    "name": "CPU Usage",
                    "cat": "resource",
                    "ph": "C",
                    "ts": timestamp,
                    "pid": 0,
                    "tid": sequential_tid,
                    "args": {
                        "Percent": float(cpu_usage)
                    }
                })

            # Match for system resource logs
            system_resource_match = re.match(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[PID: (\d+)\] - <<System>> Mem: (\d+(\.\d+)?) GB, CPU: (\d+(\.\d+)?) %",
                line)

            if system_resource_match:
                glog_timestamp, pid, sys_memory_usage, _, sys_cpu_usage, _ = system_resource_match.groups(
                )
                timestamp = parse_glog_timestamp(glog_timestamp)
                thread_id = int(pid)

                # Assign a sequential thread ID if not already assigned
                if thread_id not in thread_id_map:
                    thread_id_map[thread_id] = next_tid
                    next_tid += 1

                sequential_tid = thread_id_map[thread_id]

                # System memory usage counter event
                events.append({
                    "name": "System Memory Usage",
                    "cat": "resource",
                    "ph": "C",
                    "ts": timestamp,
                    "pid": 0,
                    "tid": sequential_tid,
                    "args": {
                        "GB": float(sys_memory_usage)
                    }
                })

                # System CPU usage counter event
                events.append({
                    "name": "System CPU Usage",
                    "cat": "resource",
                    "ph": "C",
                    "ts": timestamp,
                    "pid": 0,
                    "tid": sequential_tid,
                    "args": {
                        "Percent": float(sys_cpu_usage)
                    }
                })

            # Match for GPU logs
            gpu_resource_match = re.match(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[PID: (\d+)\] - <<GPU>> Mem: (\d+(\.\d+)?) GB, Usage: (\d+(\.\d+)?) %",
                line)

            if gpu_resource_match:
                log_timestamp, pid, gpu_memory_usage, _, gpu_usage, _ = gpu_resource_match.groups(
                )
                timestamp = parse_glog_timestamp(log_timestamp)

                # GPU memory usage counter event
                events.append({
                    "name": "GPU Memory Usage",
                    "cat": "resource",
                    "ph": "C",
                    "ts": timestamp,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "GB": float(gpu_memory_usage)
                    }
                })

                # GPU usage counter event
                events.append({
                    "name": "GPU Usage",
                    "cat": "resource",
                    "ph": "C",
                    "ts": timestamp,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "Percent": float(gpu_usage)
                    }
                })

    trace = {"traceEvents": events, "displayTimeUnit": "ns"}

    with open(output_file, "w") as f:
        json.dump(trace, f, indent=4)

    print(f"Trace file saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process log file and generate Perfetto trace JSON.")
    parser.add_argument("timer_log_file", help="Path to the timer log file")
    parser.add_argument("resource_log_file",
                        help="Path to the resource log file")
    parser.add_argument("output_file", help="Path to the output JSON file")

    args = parser.parse_args()

    process_log_file(args.timer_log_file, args.resource_log_file,
                     args.output_file)
