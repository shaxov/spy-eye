import abc
import cv2


class AbstractFilter:

    @abc.abstractmethod
    def __call__(self, frame):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass


class FakeFilter(AbstractFilter):

    def __call__(self, frame):
        return frame

    @property
    def name(self):
        return "fake_filter"

    def __repr__(self):
        return "<FakeFilter()>"


class GrayScaleFilter(AbstractFilter):

    def __call__(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @property
    def name(self):
        return "gray_scale_filter"

    def __repr__(self):
        return "<GrayScaleFilter()>"


def get(name, params=None):
    if name == 'fake_filter':
        return FakeFilter()
    elif name == 'gray_scale_filter':
        return GrayScaleFilter()
    else:
        raise ValueError(f"Frame filter with name '{name}' is not found.")


def get_list(name_params_list):
    return [get(**entry) for entry in name_params_list]


def apply(filters, image):
    for flt in filters:
        image = flt(image)
    return image
