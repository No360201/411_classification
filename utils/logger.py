import logging




class Logger():
    def __init__(self, logname, logger):
        '''
           指定保存日志的文件路径，日志级别，以及调用文件
           将日志存入到指定的文件中
        '''

        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(logname,mode='w')
        fh.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)


        # 给logger添加handler
        self.logger.addHandler(fh)


    def getlog(self):
        return self.logger