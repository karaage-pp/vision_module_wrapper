#!/usr/bin/env python
# coding: utf-8
import rospy
import numpy as np
from vision_module.msg import ObjectInfo, FaceInfo
import tf2_ros
import tf


def rotM(p):  # TODO TF関連のパッケージにする
    # 回転行列を計算する
    px = p[0]
    py = p[1]
    pz = p[2]
    # 物体座標系の 3->2->1 軸で回転させる
    Rx = np.array([[1, 0, 0],
                  [0, np.cos(px), np.sin(px)],
                  [0, -np.sin(px), np.cos(px)]])
    Ry = np.array([[np.cos(py), 0, -np.sin(py)],
                  [0, 1, 0],
                  [np.sin(py), 0, np.cos(py)]])
    Rz = np.array([[np.cos(pz), np.sin(pz), 0],
                  [-np.sin(pz), np.cos(pz), 0],
                  [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R


def calctransform(p, B, sP_B):  # TODO TF関連のパッケージにする
    # p = np.array([np.pi,np.pi/2, np.pi/3])
    R = rotM(p)
    A = R.T

    O = np.array([0, 0, 0])
    sP_O = np.dot(A, sP_B)
    rB_O = B - O
    rP_O = rB_O + sP_O
    return rP_O


class VisionWrapper():  # TODO tfの名前を可変にする。TFの名前の取得をtopicから行う
    """
    ikeda_visionの情報をワールド座標系に変換してpublishし直すためのwrapper
    object_recognition_info, face_recognition_infoのみwrappを行う
    """
    def __init__(self):
        rospy.init_node("vision_wrapper")
        self.parent_frame = rospy.get_param("~parent_frame", "map") # TODO param のネームスペース
        self.child_frame = rospy.get_param("~child_frame", "head_rgbd_sensor_rgb_frame")
        self.buf = tf2_ros.Buffer()
        self.lis = tf2_ros.TransformListener(self.buf)
        self.pub_object = rospy.Publisher(
            "/vision_module/object_recognition_info_wrapped",
            ObjectInfo, queue_size=1)

        self.pub_face = rospy.Publisher(
            "/vision_module/face_recognition_info_wrapped",
            FaceInfo, queue_size=1)

        self.pub_plane = rospy.Publisher(
            "/vision_module/plane_detection_info_wrapped",
            ObjectInfo, queue_size=1)

        self.pub_darknet = rospy.Publisher(
            "/vision_module/darknet_info_wrapped",
            ObjectInfo, queue_size=1)

        self.sub_object = rospy.Subscriber(
            "vision_module/object_recognition_info",
            ObjectInfo, self.object_info_wrapper)

        self.sub_face = rospy.Subscriber(
            "vision_module/face_detection_info",
            FaceInfo, self.face_info_wrapper)

        self.sub_plane = rospy.Subscriber(
            "vision_module/plane_detection_info",
            ObjectInfo, self.plane_info_wrapper)

        self.sub_darknet = rospy.Subscriber(
            "vision_module/object_detection_info",
            ObjectInfo, self.darknet_info_wrapper)

    def close_noed(self):
        rospy.loginfo("Have a nice day")

    def calc_object_pos(self, vector, tt):
        obj_pos_from_head = np.array([vector.x, vector.y, vector.z])
        try:
            map2head = self.buf.lookup_transform(self.parent_frame, self.child_frame, tt, rospy.Duration(3.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.sleep(1)
            rospy.logwarn("type:{0}".format(type(e)))
            rospy.logwarn("args:{0}".format(e.args))
            rospy.logwarn("message:{0}".format(e.message))
            rospy.logwarn("{0}".format(e))
            return None
        rot = tf.transformations.euler_from_quaternion([map2head.transform.rotation.x, map2head.transform.rotation.y, map2head.transform.rotation.z, map2head.transform.rotation.w])
        rot = np.array(rot)
        pos = np.array([map2head.transform.translation.x, map2head.transform.translation.y, map2head.transform.translation.z])
        obj_pos_from_map = calctransform(rot, pos, obj_pos_from_head)
        return list(obj_pos_from_map)

    def object_info_wrapper(self, objects_info):
        rospy.loginfo("object call back")
        if len(objects_info.objects) is 0:
            rospy.loginfo("object is none")
            return None
        info_time = rospy.Time(objects_info.header.stamp.secs,
                               objects_info.header.stamp.nsecs)
        for i in range(len(objects_info.objects)):
            map2object = self.calc_object_pos(
                objects_info.objects[i].camera, info_time)
            if map2object is None:
                return None
            objects_info.objects[i].camera.x = map2object[0]
            objects_info.objects[i].camera.y = map2object[1]
            objects_info.objects[i].camera.z = map2object[2]
        objects_info.header.frame_id = self.parent_frame
        rospy.loginfo("pub object")
        self.pub_object.publish(objects_info)

    def face_info_wrapper(self, face_info):
        """
        face の情報をparent_frameの座標系に変換してpublishする
        """
        if len(face_info.faces) is 0:
            return None
        info_time = rospy.Time(face_info.header.stamp.secs,
                               face_info.header.stamp.nsecs)
        for i in range(len(face_info.faces)):
            map2face = self.calc_object_pos(
                face_info.faces[i].camera, info_time)
            if map2face is None:
                return None
            face_info.faces[i].camera.x = map2face[0]
            face_info.faces[i].camera.y = map2face[1]
            face_info.faces[i].camera.z = map2face[2]
            # dumy data
            face_info.faces[i].width = 50
            face_info.faces[i].height = 50
            face_info.faces[i].age = 30
            face_info.faces[i].gender = "Male"
            face_info.faces[i].bgr = []
        face_info.header.frame_id = self.parent_frame
        self.pub_face.publish(face_info)

    def plane_info_wrapper(self, plane_info):
        """
        plane の情報をparent_frameの座標系に変換してpublishする
        """
        if len(plane_info.planes) is 0:
            return None
        # TFで変換するための時間を取得
        info_time = rospy.Time(plane_info.header.stamp.secs,
                               plane_info.header.stamp.nsecs)
        # 座標変換
        # TODO 綺麗に書きなおす
        for i in range(len(plane_info.planes)):
            # center
            map2plane = self.calc_object_pos(
                plane_info.planes[i].center, info_time)
            if map2plane is None:
                return None

            plane_info.planes[i].center.x = map2plane[0]
            plane_info.planes[i].center.y = map2plane[1]
            plane_info.planes[i].center.z = map2plane[2]

            # normal
            map2plane = self.calc_object_pos(
                plane_info.planes[i].normal, info_time)
            if map2plane is None:
                return None

            plane_info.planes[i].normal.x = map2plane[0]
            plane_info.planes[i].normal.y = map2plane[1]
            plane_info.planes[i].normal.z = map2plane[2]

            # upperLeft
            map2plane = self.calc_object_pos(
                plane_info.planes[i].upperLeft, info_time)
            if map2plane is None:
                return None

            plane_info.planes[i].upperLeft.x = map2plane[0]
            plane_info.planes[i].upperLeft.y = map2plane[1]
            plane_info.planes[i].upperLeft.z = map2plane[2]

            # lowerLeft
            map2plane = self.calc_object_pos(
                plane_info.planes[i].lowerLeft, info_time)
            if map2plane is None:
                return None

            plane_info.planes[i].lowerLeft.x = map2plane[0]
            plane_info.planes[i].lowerLeft.y = map2plane[1]
            plane_info.planes[i].lowerLeft.z = map2plane[2]

            # upperRight
            map2plane = self.calc_object_pos(
                plane_info.planes[i].upperRight, info_time)
            if map2plane is None:
                return None

            plane_info.planes[i].upperRight.x = map2plane[0]
            plane_info.planes[i].upperRight.y = map2plane[1]
            plane_info.planes[i].upperRight.z = map2plane[2]

            # lowerRight
            map2plane = self.calc_object_pos(
                plane_info.planes[i].lowerRight, info_time)
            if map2plane is None:
                return None

            plane_info.planes[i].lowerRight.x = map2plane[0]
            plane_info.planes[i].lowerRight.y = map2plane[1]
            plane_info.planes[i].lowerRight.z = map2plane[2]
            print i
        plane_info.header.frame_id = self.parent_frame
        plane_info.bgr = []
        plane_info.points = []
        plane_info.label = []
        self.pub_plane.publish(plane_info)

    def darknet_info_wrapper(self, darknet_info):
        rospy.loginfo("object call back")
        if len(darknet_info.objects) is 0:
            rospy.loginfo("object is none")
            return None
        #plane base の　detectionと分離するよう
        if len(darknet_info.objects[0].generic) is 0:
            return None
        info_time = rospy.Time(darknet_info.header.stamp.secs,
                               darknet_info.header.stamp.nsecs)
        for i in range(len(darknet_info.objects)):
            map2object = self.calc_object_pos(
                darknet_info.objects[i].camera, info_time)
            if map2object is None:
                return None
            darknet_info.objects[i].camera.x = map2object[0]
            darknet_info.objects[i].camera.y = map2object[1]
            darknet_info.objects[i].camera.z = map2object[2]
            darknet_info.objects[i].bgr = []
        darknet_info.header.frame_id = self.parent_frame
        darknet_info.bgr = []
        darknet_info.points = []
        rospy.loginfo("pub object")
        self.pub_darknet.publish(darknet_info)

if __name__ == '__main__':
    vision_wrapper = VisionWrapper()
    rospy.loginfo("Vision wrapper has starting")
    rospy.spin()
