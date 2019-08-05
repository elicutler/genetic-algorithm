import logging

class DefaultLogger:
    defaultFormatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(module)s:%(lineno)s] - %(message)s',
        datefmt="%Y-%m-%d %H:%M"
    )
    '''
    Set up root logger with customizable default stream and file handlers
    -----
    params
        logFileName: name of file to log output to
        level: logging level for root logger and handlers
        defaultFormatterOverride: optional logging formatter to override default
        setDefaultStreamHandler: whether to set default stream handler 
        setDefaultFileHandler: whether to set default file handler
    public methods
        none
    public attributes
        :self.logger: root logger, set with defaults
    '''
    def __init__(
        self, 
        logFileName: str, 
        level: int = logging.INFO, 
        defaultFormatterOverride: logging.Formatter = None,
        setDefaultStreamHandler: bool = True, 
        setDefaultFileHandler: bool = True,
        defaultFileHandlerMode: str = 'w'
    ) -> None:
        self.logFileName = logFileName
        self.level = level
        self.defaultFormatterOverride = defaultFormatterOverride
        self.setDefaultStreamHandler = setDefaultStreamHandler
        self.setDefaultFileHandler = setDefaultFileHandler
        self.defaultFileHandlerMode = defaultFileHandlerMode
        
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level)
        
        self.formatter = self._setFormatter()
            
        if self.setDefaultStreamHandler:
            streamHandler = self._setDefaultHandler(type_='stream')
            self.logger.addHandler(streamHandler)
            
        if self.setDefaultFileHandler:
            fileHandler = self._setDefaultHandler(type_='file')
            self.logger.addHandler(fileHandler)
        return None
            
    def _setFormatter(self) -> logging.Formatter:
        if self.defaultFormatterOverride is None:
            formatter = self.defaultFormatter
        else: 
            formatter = self.defaultFormatterOverride
        return formatter
    
    def _setDefaultHandler(self, type_: str) -> logging.Handler:
        if type_ == 'stream':
            handler = self._setDefaultStreamHandler()
        elif type_ == 'file':
            handler = self._setDefaultFileHandler()
        self._setHandlerDefaults(handler)
        return handler
    
    @staticmethod
    def _setDefaultStreamHandler() -> logging.StreamHandler:
        streamHandler = logging.StreamHandler()
        return streamHandler
    
    def _setDefaultFileHandler(self) -> logging.FileHandler:
        fileHandler = logging.FileHandler(
            self.logFileName, self.defaultFileHandlerMode
        )
        return fileHandler
    
    def _setHandlerDefaults(self, handler: logging.Handler) -> None:
        handler.setLevel(self.level)
        handler.setFormatter(self.formatter)
        return None    
