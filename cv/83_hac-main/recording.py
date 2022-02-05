import cv2
import time
import keyboard
import traceback
import sys
import threading
import argparse
import os

from pyhac import hac

record = False
end_record = False
t = None

def timer(record_time):
    global end_record
    time.sleep(record_time)
    end_record = True
    print("End", end_record)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='test')
    parser.add_argument('-k', '--keep_data', type=str2bool, default="True")
    args = parser.parse_args()

    record_time = 30

    cap = cv2.VideoCapture(0)
    factor = 1080 / 1920
    ui_factor = 1
    width = int(1920//2/ui_factor)
    height = width * factor
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    csv_path = os.path.join("data", "actions", args.dataset, "data.csv")
    image_dir = os.path.join("data", "actions", args.dataset, "image")

    count = 0                         
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():

        try:
            s = time.time()
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # cap.get(cv2.CAP_PROP_POS_MSEC) can be zero
            # https://stackoverflow.com/questions/44759407/why-does-opencv-cap-getcv2-cap-prop-pos-msec-only-return-0
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            if abs(ts - 0.0) < 1e-8:
                ts = count / fps

            hac.update(image, ts, keep_data=args.keep_data)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            hac.holistic_tracker.draw_landmarks(image)
            cv2.imshow('HAC demo', cv2.flip(image, 1))
            cv2.moveWindow('HAC demo', 0, 0)
            
            e = time.time()
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

            if record:
                if t is None:                
                    t = threading.Thread(target=timer, daemon=True, args=(record_time,))
                    t.daemon = True
                    t.start()
                hac.save(csv_path, image_dir, 0)

            if end_record:
                break

            if keyboard.is_pressed("r") and not record:
                print("Start to record after 3 seconds...")
                record = True
                time.sleep(3)
            
        except Exception as e:
            traceback.print_exc()
            break

        count += 1
    
    cap.release()
    cv2.destroyAllWindows()

    if args.keep_data:
        print("Saving images...")
        hac.save_images()

    print("Finish!")

