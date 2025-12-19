from typing import List

from globals import MW


class DataRead:
    def __init__(self, exts: tuple[str, ...] = ('avi', 'mp4', 'mkv')):
        self.exts = exts
        self.video_path_list: List[str] = []
        # 初始化时立即扫描
        self.scan()

    # ------------- 对外 API -------------
    def scan(self) -> List[str]:
        """扫描/刷新缓存，返回绝对路径列表"""
        self.video_path_list = [
            str(p.resolve())
            for ext in self.exts
            for p in MW.PARAMS_MAP['data_dir'].rglob(f'*.{ext}')
        ]
        if not self.video_path_list:
            print(f"[WARN] 在 {MW.PARAMS_MAP['data_dir']} 未找到任何 {self.exts} 视频")
        return self.video_path_list

    # 缓存_video_path_list
    @property
    def get_video_path_list(self) -> List[str]:
        """第一次访问时自动扫描，后续直接返回缓存"""
        if not self.video_path_list:  # 延迟加载
            self.scan()
        return self.video_path_list
