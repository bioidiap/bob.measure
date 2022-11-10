#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy

from bob.measure.utils import confidence_for_indicator_variable


def test_confidence_interval():
    def assert_confidence(x, n, expected_lower, expected_upper):
        lower, upper = confidence_for_indicator_variable(x, n)
        assert numpy.allclose(lower, expected_lower)
        assert numpy.allclose(upper, expected_upper)

    assert_confidence(1, 2, 0.01257911709342505, 0.98742088290657493)
    assert_confidence(10, 10, 0.69150289218123917, 1)
    assert_confidence(0, 10, 0, 0.30849710781876077)
