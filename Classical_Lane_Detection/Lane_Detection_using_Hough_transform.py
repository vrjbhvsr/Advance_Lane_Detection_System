import cv2
import numpy as np

def draw_lines(image, lines):
    line_image = np.zeros((image.shape[0], image.shape[1], 3), 'uint8')
    if lines is not None:
        for line in lines:
            if len(line[0]) == 4:  # Ensure that the line has exactly 4 values
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
    image_with_lines = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return image_with_lines

        
def image_mask(image, ROI):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, ROI,255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def detect_lanes(image):
    height, width =image.shape[0:2]
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(gray_image,(5,5),1)
    canny= cv2.Canny(blur,50,150)
    region_of_interest_vertices = np.array([[(0, height), (width / 2, height * 0.65), (width, height)]],np.int32)
    cropped_image= image_mask(canny,region_of_interest_vertices)
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40,
                           maxLineGap=150
                           )
    print(lines)
    image_with_lines = draw_lines(image, lines)
    return image_with_lines


cap = cv2.VideoCapture(r"C:\Users\49179\Desktop\computer_vision_course_materials\lane_detection_video.mp4")
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_lanes(frame)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
