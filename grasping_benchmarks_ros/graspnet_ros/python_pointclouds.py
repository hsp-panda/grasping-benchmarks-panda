#!/usr/bin/env python
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct

fmt_full = ''

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

_NP_TYPES = {
    np.dtype('uint8')   :   (PointField.UINT8,  1),
    np.dtype('int8')    :   (PointField.INT8,   1),
    np.dtype('uint16')  :   (PointField.UINT16, 2),
    np.dtype('int16')   :   (PointField.INT16,  2),
    np.dtype('uint32')  :   (PointField.UINT32, 4),
    np.dtype('int32')   :   (PointField.INT32,  4),
    np.dtype('float32') :   (PointField.FLOAT32,4),
    np.dtype('float64') :   (PointField.FLOAT64,8)
}

def pointcloud2_to_array(msg):
    global fmt_full
    if not fmt_full:
        fmt = _get_struct_fmt(msg)
        fmt_full = '>' if msg.is_bigendian else '<' + fmt.strip('<>')*msg.width*msg.height
    # import pdb; pdb.set_trace()
    unpacker = struct.Struct(fmt_full)
    unpacked = np.asarray(unpacker.unpack_from(msg.data))
    unpacked = unpacked.reshape(msg.height, msg.width, len(msg.fields))
    unpacked = unpacked[:,:,0:3]
    return unpacked
    # unpacked.reshape(msg.height, msg.width, len(msg.fields))

def _get_struct_fmt(cloud, field_names=None):
    fmt = '>' if cloud.is_bigendian else '<'
    offset = 0
    for field in (f for f in sorted(cloud.fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print >> sys.stderr, 'Skipping unknown PointField datatype [%d]' % field.datatype
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

rate_pub = None

def cloud_cb(msg):
    pointcloud2_to_array(msg)
    rate_pub.publish()

if __name__ == '__main__':
    import rospy
    from std_msgs.msg import Empty
    rospy.init_node('test_pointclouds')
    rate_pub = rospy.Publisher('rate', Empty)
    rospy.Subscriber('/camera/depth_registered/points', PointCloud2, cloud_cb)
    rospy.spin()