#!/usr/bin/env python
#
#   Copyright (C) 2012  Don Smiley  ds@sidorof.com

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

#   See the LICENSE file included in this archive
#

"""
This module implements some basic utilities for use with Grammatical Evolution

"""

from random import random


def rand_weight(constraint=1.0):
    """
    Returns a random weight centered around 0.  The constrain limits
    the value to the maximum that it would return.  For example, if
    .5 is the constraint, then the returned weight would between -.5 and +.5.
    """
    return random() * 2.0 * constraint - constraint


def base10tobase2(value, zfill=0):
    """
    This function converts from base 10 to base 2 in string format.  In
    addition, it takes a parameter for zero filling to pad out to a specific
    length.

    Note that the incoming number is converted to an int, and that if it is
    a negative number that the negative sign is added on to the total length,
    resulting in a string 1 char longer than the zfill specified.

    """
    new_value = []
    val = int(value)
    if val < 0:
        neg = True
        val *= -1
    else:
        neg = False

    if val == 0:
        new_value = ['0']

    while val > 0:
        new_value.append(str(val % 2))
        val = val / 2

    new_value.reverse()
    new_value_str = ''.join(new_value)
    if zfill:
        if len(new_value_str) > zfill:
            raise ValueError("""
            Base 2 version of %s is longer, %s, than the zfill limit, %s
            """ % (value, new_value_str, zfill))
        else:
            new_value_str = new_value_str.zfill(zfill)

    if neg:
        new_value_str = "-" + new_value_str

    return new_value_str


def base2tobase10(value):
    """
    This function converts from base 2 to base 10.  Unlike base10tobase2,
    there is no zfill option, and the result is output as an int.

    """

    new_value = 0
    val = str(value)
    if val < 0:
        neg = True
        val *= -1
    else:
        neg = False

    val = str(value)

    factor = 0
    for i in range(len(val) - 1, -1, -1):
        if not val[i] == '-':
            new_value += int(val[i]) * pow(2, factor)
        else:
            neg = True
        factor += 1

    if neg:
        new_value *= -1

    return new_value
