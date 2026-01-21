import os
import sys
import logging
import datetime


ROOT_DIR = os.getcwd()
LOG_DIR = os.path.join(ROOT_DIR, "logs")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)


def clean_old_logs(log_dir, max_files=5):
    """清理旧的日志文件，保留最新的 max_files 个文件"""
    log_files = []
    for filename in os.listdir(log_dir):
        if filename.startswith("app_") and filename.endswith(".log"):
            file_path = os.path.join(log_dir, filename)
            try:
                # 提取时间戳
                timestamp_str = filename.split("_")[1].split(".")[0]
                datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                log_files.append((os.path.getctime(file_path), file_path))
            except (IndexError, ValueError):
                continue

    # 按创建时间排序（旧到新）
    log_files.sort(key=lambda x: x[0])

    # 删除超出数量的旧文件
    if len(log_files) >= max_files:
        files_to_delete = log_files[:-max_files]
        for _, file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"已删除旧日志文件: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"删除日志文件失败 {file_path}: {e}")


class Logger:
    def __init__(self, name = "multimodal_rag"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # 清除已有的处理器

        # 控制台输出格式
        console_formatter = logging.Formatter(
            '%(asctime)s | %(processName)s | %(threadName)s | %(module)s.%(funcName)s:%(lineno)d | %(levelname)s: %(message)s',
            datefmt='%Y%m%d %H:%M:%S'
        )

        # 文件输出格式
        file_formatter = logging.Formatter(
            '%(asctime)s | %(processName)s | %(threadName)s | %(module)s.%(funcName)s:%(lineno)d | %(levelname)s: %(message)s',
            datefmt='%Y%m%d %H:%M:%S'
        )

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 生成带时间戳的日志文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"app_{timestamp}.log"
        log_file_path = os.path.join(LOG_DIR, log_file)

        # 文件处理器（覆盖写入，不追加）
        file_handler = logging.FileHandler(
            log_file_path,
            encoding='utf-8',
            mode='w'  # 覆盖写入模式
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 清理旧日志文件
        clean_old_logs(LOG_DIR, max_files=5)

    def get_logger(self):
        return self.logger


# 全局单例
logger = Logger().get_logger()

if __name__ == "__main__":
    # 测试日志功能
    logger.debug("这是一个调试信息。")
    logger.info("这是一个普通信息。")
    logger.warning("这是一个警告信息。")
    logger.error("这是一个错误信息。")
    logger.critical("这是一个严重错误信息。")
    logger.info(f"日志文件存储在: {LOG_DIR}")

    # 获取当前日志文件名（从handler中获取）
    log_file_path = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file_path = handler.baseFilename
            break

    if log_file_path:
        logger.info(f"日志文件路径: {log_file_path}")
        logger.info(f"日志文件名: {os.path.basename(log_file_path)}")
    logger.info(f"根目录路径: {ROOT_DIR}")
