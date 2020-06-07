import win32api, win32con, win32gui

class MainWindow:

    def __init__(self):
        win32gui.InitCommonControls()
        self.hinst = win32api.GetModuleHandle(None)
        className = 'MainWindow'
        message_map = {
            win32con.WM_DESTROY: self.OnDestroy,
        }
        wc = win32gui.WNDCLASS()
        wc.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
        wc.hbrBackground = win32gui.GetStockObject(win32con.DKGRAY_BRUSH)
        wc.lpfnWndProc = message_map
        wc.lpszClassName = className
        win32gui.RegisterClass(wc)
        style = win32con.WS_OVERLAPPEDWINDOW
        self.hwnd = win32gui.CreateWindow(
            className,
            'GazeTracking',
            style,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            800,
            600,
            0,
            0,
            self.hinst,
            None
        )
        win32gui.UpdateWindow(self.hwnd)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)

    def OnDestroy(self, hwnd, message, wparam, lparam):
        win32gui.PostQuitMessage(0)
        return True