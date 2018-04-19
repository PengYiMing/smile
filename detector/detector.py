import cv2 as cv
import numpy as np
import tensorflow as tf

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

if __name__ == '__main__':
    face_detector = cv.CascadeClassifier('./pre-trained/haarcascade_frontalface_default.xml')
    smile_detector = cv.CascadeClassifier('./pre-trained/haarcascade_smile.xml')

    tf.reset_default_graph()
    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)
    saver = tf.train.import_meta_graph('../cnn/smile-model-300.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../cnn/'))
    X_holder = graph.get_tensor_by_name('X_holder:0')
    Y_predict = graph.get_tensor_by_name('Y_predict:0')

    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        # detect face
        ret, frame = capture.read()
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.2,\
                minNeighbors=5, minSize=(32, 32))
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), color=BLUE)
            # detect smile in face
            face = frame[y:y + h, x:x + w]
            gray_face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            smiles = smile_detector.detectMultiScale(gray_face, scaleFactor=1.8,\
                    minNeighbors=20, minSize=(48, 24))
            for sx, sy, sw, sh in smiles:
                cv.rectangle(face, (sx, sy), (sx + sw, sy + sh), color=GREEN)
                # verify smile
                smile = face[sy:sy + sh, sx:sx + sw]
                gray_smile = cv.cvtColor(smile, cv.COLOR_BGR2GRAY)
                resized_smile = cv.resize(gray_smile, (64, 32), interpolation=cv.INTER_NEAREST)
                cv.imshow('debug', resized_smile)
                resized_smile = resized_smile[np.newaxis, :, :, np.newaxis] / 255
                y_predict = sess.run(Y_predict, feed_dict={X_holder: resized_smile})
                chance = np.squeeze(y_predict)
                if chance >= 0.6:
                    cv.putText(face, 'smile chance: %.2f%%' % (chance * 100), (16, 16), cv.FONT_HERSHEY_PLAIN, 1.0, RED)
                else:
                    cv.putText(face, 'no smile', (16, 16), cv.FONT_HERSHEY_PLAIN, 1.0, RED)
        cv.imshow('frame', frame)

        key = cv.waitKey(50)
        if key == 32:
            break

    sess.close()
    capture.release()
    cv.destroyAllWindows()

