
import logging
import sys

from typing import Optional, Union


def setLoggerDefaults(
    logger:logging.Logger, level:int=logging.INFO, logFileName:Optional[str]=None
) -> None:
    logger.setLevel(level)
    logger.handlers = []
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(module)s:%(lineno)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    def setHandlerDefaults(handler:Union[logging.StreamHandler, logging.FileHandler]) -> None:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    streamHandler = logging.StreamHandler(sys.stdout)
    setHandlerDefaults(streamHandler)
    
    if logFileName is not None:
        fileHandler = logging.FileHandler(logFileName, 'w')
        setHandlerDefaults(fileHandler)