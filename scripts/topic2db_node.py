#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import psycopg2
import psycopg2.extras

import rospy
from vision_module.msg import ObjectInfo, FaceInfo


class Topic2DataBase():
    """
    topicの情報をdatabaseに保存するノード
    topicごとにsubscriberがいてDataBaseに情報を確保する
    """

    def __init__(self):
        rospy.init_node("topic2db_node")
        # postgresqlとの接続用
        dbname = rospy.get_param("/db_param/dbname", "")
        host = rospy.get_param("/db_param/host", "")
        user = rospy.get_param("/db_param/user", "")
        password = rospy.get_param("/db_param/password", "")
        # postgresqlと接続
        self.conn = psycopg2.connect(
            """
            dbname={0}
            host={1}
            user={2}
            password={3}
            """.format(dbname, host, user, password))
        # ObjectInfo
        self._sub_object = rospy.Subscriber(
            "vision_module/object_recognition_info_wrapped", ObjectInfo,
            self.object2database)
        # FaceInfo
        self._sub_face = rospy.Subscriber(
            "vision_module/face_recognition_info_wrapped", FaceInfo,
            self.face2database)
        # PlaneInfo
        self._sub_plane = rospy.Subscriber(
            "vision_module/plane_detection_info_wrapped", ObjectInfo,
            self.plane2database)
        # 終了時にpostgresqlとの接続を切る
        rospy.on_shutdown(self.close_psql)

    def close_psql(self):
        self.conn.close()

    def object2database(self, object_data):
        """
        object_recognition_infoをDataBaseに追加するメソッド
        """
        rostime = rospy.Time(object_data.header.stamp.secs, object_data.header.stamp.nsecs)
        for i in range(len(object_data.objects)):
            position_x = object_data.objects[i].camera.x
            position_y = object_data.objects[i].camera.y
            position_z = object_data.objects[i].camera.z
            if object_data.objects[i].generic[0].id is not -1:
                generic_id_0 = object_data.objects[i].generic[0].id
                generic_name_0 = object_data.objects[i].generic[0].name
                generic_score_0 = object_data.objects[i].generic[0].score
                generic_id_1 = object_data.objects[i].generic[1].id
                generic_name_1 = object_data.objects[i].generic[1].name
                generic_score_1 = object_data.objects[i].generic[1].score
                generic_id_2 = object_data.objects[i].generic[2].id
                generic_name_2 = object_data.objects[i].generic[2].name
                generic_score_2 = object_data.objects[i].generic[2].score
            else: #personの場合
                generic_id_0 = object_data.objects[i].generic[0].id
                generic_name_0 = object_data.objects[i].generic[0].name
                generic_score_0 = object_data.objects[i].generic[0].score
                generic_id_1 = None
                generic_name_1 = None
                generic_score_1 = None
                generic_id_2 = None
                generic_name_2 = None
                generic_score_2 = None
            specific_id_0 = object_data.objects[i].specific[0].id
            specific_name_0 = object_data.objects[i].specific[0].name
            specific_score_0 = object_data.objects[i].specific[0].score
            specific_id_1 = object_data.objects[i].specific[1].id
            specific_name_1 = object_data.objects[i].specific[1].name
            specific_score_1 = object_data.objects[i].specific[1].score
            specific_id_2 = object_data.objects[i].specific[2].id
            specific_name_2 = object_data.objects[i].specific[2].name
            specific_score_2 = object_data.objects[i].specific[2].score
            color = object_data.objects[i].color
            ros_timestamp = rostime.to_nsec()
            width = object_data.objects[i].width
            height = object_data.objects[i].height
            # bgr = object_data.objects[i].bgr
            bgr = None
            # featuresがpublishされていない時用
            if object_data.objects[i].features is not None:
                features = list(object_data.objects[i].features)
            else:
                features = None
            with self.conn:
                with self.conn.cursor() as curs:
                    curs.execute("""INSERT INTO
                                 object_info (position_x, position_y, position_z, generic_id_0, generic_name_0, generic_score_0, generic_id_1, generic_name_1, generic_score_1, generic_id_2, generic_name_2, generic_score_2, specific_id_0, specific_name_0, specific_score_0, specific_id_1, specific_name_1, specific_score_1, specific_id_2, specific_name_2, specific_score_2, color, ros_timestamp, width, height, bgr, features)
                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                                 (position_x, position_y, position_z, generic_id_0, generic_name_0, generic_score_0, generic_id_1, generic_name_1, generic_score_1, generic_id_2, generic_name_2, generic_score_2, specific_id_0, specific_name_0, specific_score_0, specific_id_1, specific_name_1, specific_score_1, specific_id_2, specific_name_2, specific_score_2, color, ros_timestamp, width, height, bgr, features))

    def face2database(self, face_data):
        """
        face_recognition_infoをDataBaseに追加するメソッド
        """
        rostime = rospy.Time(face_data.header.stamp.secs, face_data.header.stamp.nsecs)
        for i in range(len(face_data.faces)):
            position_x = face_data.faces[i].camera.x
            position_y = face_data.faces[i].camera.y
            position_z = face_data.faces[i].camera.z
            # if face_data.faces[i].generic[0].id is not -1:
            #     generic_id_0 = face_data.faces[i].generic[0].id
            #     generic_name_0 = face_data.faces[i].generic[0].name
            #     generic_score_0 = face_data.faces[i].generic[0].score
            #     generic_id_1 = face_data.faces[i].generic[1].id
            #     generic_name_1 = face_data.faces[i].generic[1].name
            #     generic_score_1 = face_data.faces[i].generic[1].score
            #     generic_id_2 = face_data.faces[i].generic[2].id
            #     generic_name_2 = face_data.faces[i].generic[2].name
            #     generic_score_2 = face_data.faces[i].generic[2].score
            # else: #personの場合
            #     generic_id_0 = face_data.faces[i].generic[0].id
            #     generic_name_0 = face_data.faces[i].generic[0].name
            #     generic_score_0 = face_data.faces[i].generic[0].score
            #     generic_id_1 = None
            #     generic_name_1 = None
            #     generic_score_1 = None
            #     generic_id_2 = None
            #     generic_name_2 = None
            #     generic_score_2 = None
            # specific_id_0 = face_data.faces[i].specific[0].id
            # specific_name_0 = face_data.faces[i].specific[0].name
            # specific_score_0 = face_data.faces[i].specific[0].score
            # specific_id_1 = face_data.faces[i].specific[1].id
            # specific_name_1 = face_data.faces[i].specific[1].name
            # specific_score_1 = face_data.faces[i].specific[1].score
            # specific_id_2 = face_data.faces[i].specific[2].id
            # specific_name_2 = face_data.faces[i].specific[2].name
            # specific_score_2 = face_data.faces[i].specific[2].score
            age = face_data.faces[i].age
            gender = face_data.faces[i].gender
            ros_timestamp = rostime.to_nsec()
            width = face_data.faces[i].width
            height = face_data.faces[i].height
            # bgr = face_data.faces[i].bgr
            bgr = None
            with self.conn:
                with self.conn.cursor() as curs:
                    curs.execute("""INSERT INTO
                                 face_info (position_x, position_y, position_z, age, gender, ros_timestamp, width, height, bgr)
                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                                 (position_x, position_y, position_z, age, gender, ros_timestamp, width, height, bgr))


    def plane2database(self, plane_data): # TODO 編集途中
        """
        plane_infoをDataBaseに追加するメソッド
        """
        rostime = rospy.Time(plane_data.header.stamp.secs, plane_data.header.stamp.nsecs)
        for i in range(len(plane_data.planes)):
            center_x = plane_data.planes[i].center.x
            center_y = plane_data.planes[i].center.y
            center_z = plane_data.planes[i].center.z
            normal_x = plane_data.planes[i].normal.x
            normal_y = plane_data.planes[i].normal.y
            normal_z = plane_data.planes[i].normal.z
            upperLeft_x = plane_data.planes[i].upperLeft.x
            upperLeft_y = plane_data.planes[i].upperLeft.y
            upperLeft_z = plane_data.planes[i].upperLeft.z
            lowerLeft_x = plane_data.planes[i].lowerLeft.x
            lowerLeft_y = plane_data.planes[i].lowerLeft.y
            lowerLeft_z = plane_data.planes[i].lowerLeft.z
            upperRight_x = plane_data.planes[i].upperRight.x
            upperRight_y = plane_data.planes[i].upperRight.y
            upperRight_z = plane_data.planes[i].upperRight.z
            lowerRight_x = plane_data.planes[i].lowerRight.x
            lowerRight_y = plane_data.planes[i].lowerRight.y
            lowerRight_z = plane_data.planes[i].lowerRight.z
            ros_timestamp = rostime.to_nsec()
            with self.conn:
                with self.conn.cursor() as curs:
                    curs.execute("""INSERT INTO
                                 plane_info (center_x, center_y, center_z, normal_x, normal_y, normal_z, upperLeft_x, upperLeft_y, upperLeft_z, lowerLeft_x, lowerLeft_y, lowerLeft_z, upperRight_x, upperRight_y, upperRight_z, lowerRight_x, lowerRight_y, lowerRight_z, ros_timestamp)
                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                                 (center_x, center_y, center_z, normal_x, normal_y, normal_z, upperLeft_x, upperLeft_y, upperLeft_z, lowerLeft_x, lowerLeft_y, lowerLeft_z, upperRight_x, upperRight_y, upperRight_z, lowerRight_x, lowerRight_y, lowerRight_z, ros_timestamp))


if __name__ == '__main__':
    topic2db = Topic2DataBase()
    rospy.loginfo("Topic2DataBase has starting")
    rospy.spin()
