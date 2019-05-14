# encoding:utf-8


class Selector(object):
    """模型选择器"""

    _support_models = {}
    _default_settings = {}
    _support_settings = {}

    def __init__(self,  **setting):
        # 真实的模型内核
        self._model = None
        # 选择器设置
        self._settings = setting

    def exe(self, *args, **kwargs):
        """使用settting构建model"""

        model = self._settings.get('model') if self._settings.get('model') \
            else self._default_settings.get('model')
        model_func = self._support_models[model]
        self._model = getattr(self, model_func)()
        target_func = self._settings.get('target') if self._settings.get('target') \
            else self._default_settings.get('target')
        return getattr(self._model, target_func)(*args, **kwargs)

    def _check_setting(self):
        for setting in self._settings.keys():
            if setting not in self._support_settings.keys():
                print('WARING: {} not support {} setting'.format(self, setting))



