import datetime
import logging
import os
from progress.bar import Bar


#继承Bar类
class ProgressBar(Bar):
    message = 'Loading'
    fill = '='
    suffix = '%(percent).1f%% | Elapsed: %(elapsed)ds | ETA: %(eta)ds '

loggers = {}
def getLogger(name, log_dir='logs/'):
    global loggers
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # file output
            fh = logging.FileHandler(
                os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")+'.txt')
            )
            #设置日志级别，大于等于DEBUG级别的日志才会被输出
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            # terminal output
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        loggers[name] = logger
        return logger
