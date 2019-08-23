from typing import Union
import logging

class DefaultLogger(logging.RootLogger):
    '''
    Set up root logger with customizable default stream and file handlers
    -----
    
    params
        logFileName -- name of file to log output to
        level -- logging level for root logger and handlers
        defaultFormatterOverride -- optional logging formatter to override default
        setDefaultStreamHandler -- whether to set default stream handler 
        setDefaultFileHandler -- whether to set default file handler
        
    public methods
        none
        
    public attributes
        logger -- root logger, set with defaults
    '''
    
    defaultFormatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(module)s:%(lineno)s] - %(message)s',
        datefmt="%Y-%m-%d %H:%M"
    )
    
    def __init__(
        self, level:int=logging.INFO, 
        defaultFormatterOverride:logging.Formatter=None,
        useDefaultStreamHandler:bool=True, 
        logFileName:Union[str, None]=None, defaultFileHandlerMode:str='w'
    ) -> None:
        super().__init__(level)
        
        self.defaultFormatterOverride = defaultFormatterOverride
        self.useDefaultStreamHandler = useDefaultStreamHandler
        self.logFileName = logFileName
        self.defaultFileHandlerMode = defaultFileHandlerMode
        
        self.formatter = self._setFormatter()
        
        logging.handlers = []
            
        if self.useDefaultStreamHandler:
            streamHandler = self._setDefaultHandler(type_='stream')
            self.addHandler(streamHandler)
            
        if self.logFileName is not None:
            fileHandler = self._setDefaultHandler(type_='file')
            self.addHandler(fileHandler)
            
    def _setFormatter(self) -> logging.Formatter:
        if self.defaultFormatterOverride is None:
            formatter = self.defaultFormatter
        else: 
            formatter = self.defaultFormatterOverride
        return formatter
    
    def _setDefaultHandler(self, type_:str) -> logging.Handler:
        if type_ == 'stream':
            handler = self._setDefaultStreamHandler()
        elif type_ == 'file':
            handler = self._setDefaultFileHandler()
        self._setHandlerDefaults(handler)
        return handler

    def _setDefaultFileHandler(self) -> logging.FileHandler:
        fileHandler = logging.FileHandler(
            self.logFileName.strip('.py') + '.log', 
            self.defaultFileHandlerMode
        )
        return fileHandler
    
    @staticmethod
    def _setDefaultStreamHandler() -> logging.StreamHandler:
        streamHandler = logging.StreamHandler()
        return streamHandler
    
    def _setHandlerDefaults(self, handler:logging.Handler) -> None:
        handler.setLevel(self.level)
        handler.setFormatter(self.formatter)   
