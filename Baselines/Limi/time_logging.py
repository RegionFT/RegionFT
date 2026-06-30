import time


class TimeLogger:
    def __init__(self, file='log.txt', label='result'):
        self.start_cpu_time = None
        self.end_cpu_time = None
        self.start_real_time = None
        self.end_real_time = None
        self.log_file = file
        self.log_label = label

    def start_point(self):
        self.start_cpu_time = time.process_time()
        self.start_real_time = time.time()

    def end_point(self, need_log_time=True, add_res=None):
        self.end_cpu_time = time.process_time()
        self.end_real_time = time.time()
        if need_log_time:
            self.log_time(add_res)

    def log_time(self, add_res=None):
        cpu_time_elapsed = self.end_cpu_time - self.start_cpu_time
        real_time_elapsed = self.end_real_time - self.start_real_time

        if add_res is not None and isinstance(add_res, str):
            add_res_str = ',' + add_res
        else:
            add_res_str = ''
        log_message = (
            f"{self.log_label},{cpu_time_elapsed:.2f},{real_time_elapsed:.2f}{add_res_str}\n"
        )

        with open(self.log_file, 'a') as log_file:
            log_file.write(log_message)
        print("Time logged successfully")
