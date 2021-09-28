# lets just go through what will happen in the motion_detection.
'''
1) first we will have a background image . when cam is first triggered. we can use this as a base image and compare this image with other images
2) lets say after some time someone apears on camera. lets think it as current frame.
3) convert background and current image to gray scale but not directly convert it . first assign backgorund to some variable and then convert it
4)after assigning both of them to a variable and converting them to greyscale, take look for difference between them
5) call threshold - that is if diff is say > 1000 convert those pixels to white else covert them to black
6)next find countours(boundary or shape) - outline of the face for example.
7)then use for loop to iterate through all the countours. if area of contour is more than 500 pixels -> consider it as an moving object , else non moving object.
8) we will draw rectangles that are greater than the min area and then we will show the rectangles in our current image
9)and then we collect the time of moving object entering and leaving the background



'''
import cv2
# needed when we want to access current time so that we could store the time of an object being entered and left
from datetime import datetime

import pandas  # to create data frame that holds the time list

# we need a way to store the image as soon as the cam starts and we dont wan't that value to change .so assign it to a variable (***the first iteration will yield background image***) . so i'm assigning it to a none value. after first iteration we will get the background image

first_frame = None

# why should we put none , none ??
# beacuse list has no items so far and when code executes it needs to compare the last two elements of the index remember . so we put none values for the last two index

# we want to know when the object is enter -> status changing from 0 to 1 and also when object left . status changing from 1 to zero
status_list = [None, None]

# to read a video from webcam or from alreay captured vid. numer represent the cam - say web cam (0). external cam (1). if i want to access vid file i need to put name of the file

times = []  # creating this to record how mant=y times status has changed

# created dtataframe that holds two colums
df = pandas.DataFrame(columns=["Start", "End"])


video = cv2.VideoCapture(0)

# to show vid put the code inside while loop . this will execute code infinitely as long as cam is on . so we get image in infinite nu7m of times so we get videos

while True:

    # first our cam needs to searc for a frame so im creating read method
    check, frame = video.read()

    status = 0  # No motion

    # printing coordinates and colurs of my face - op will be numpy array

    # print(check)
    # print(frame)

    # converting to grey scale version
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gives blurry version of the image passed . increases accuracy and removes noise .
    # and another parameter is about the parameter of blurryness and its must be in a tuple
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # this will get executed only in first iteration.assigns background image to a grey version
    if first_frame is None:
        first_frame = gray
        continue         # as soon as if executes we dont need the rest of the loop to be executed because we need the background to original . so im putting continue so that it will go back to start of the loop

    # after first iteration now in the second iteration if is not executed . now i have two greyscales. now i can find the diff between the two
    # taking diff of two blurried images
    delta_frame = cv2.absdiff(first_frame, gray)

    # so what we are doing here is checking where the difference is greater than 30 , and converting those pixels to White(255 value gives white) , and adding a threshold method
    # (thresh binary returns a tuple . but we need only the second index that is returned . i.e delta-frame . so [1])
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # dilate method basically smothens whichever frame passed inside . removes blcak paches which are inside the white frames.
    # None is given beacuse there is no array to pass.
    # iterates 2 times for smoother
    thresh_frame = cv2.dilate(thresh_delta, None, iterations=2)

    # now we have to find the boundry or contours. fo that we have a method - (cnts,_)
    (cnts, _) = cv2.findContours(thresh_frame.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # this gives many boundrys but i need to iterate through that and gind if the boundry area < 1000 pixcels go to the next contour. check again and again
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        # when we get cange in more than 10000 pixcel , that means an object is moving , status = 1 (motion)
        status = 1

        # x y w h are partameters of rectangle, if area is larger than 1000 pixcel , give us rectangular border
        (x, y, w, h) = cv2.boundingRect(contour)
        # {1) x,y is the first coordinate of the face}   {2) second param is the lower right corner - x+width and y + height }    {3) (0, 255, 0) colour - rectangle will be green in colour }    { 4) 3 is width}
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    status_list.append(status)

    # so here we are checking if there is any change in status in last two elements of the status lis . if yes it will tell us that a mooving object just entered the range of the camera
    if status_list[-1] == 1 and status_list[-2] == 0:  # this gives entering time

        times.append(datetime.now())

    # this gives leaving time(1 to zeop)
    if status_list[-1] == 0 and status_list[-2] == 1:

        times.append(datetime.now())

        # cv2.imshow("gray Frame", gray)  # shows blurried image
    cv2.imshow("delta Frame", delta_frame)  # shows blurried image
    cv2.imshow("thresh_delta", thresh_delta)

    cv2.imshow("smoother_frames", thresh_frame)

    cv2.imshow("contour_frame", frame)

    # each image will be captured 0.001ms

    key = cv2.waitKey(1)

    # pressing q button will break the program and cam will be shutdown

    if key == ord('q'):
        if status == 1:
            # why did i write this?? because if i quit the program when moving or if i quit the program when status == 1 , then we only get entering time and dont have leaving time . so i must create a exit time by force
            times.append(datetime.now())
        break
    # print(status)

# release the camera
print(status_list)  # outside the loop beacuse i dont want to print each time. this will give complete detail about object entering and leaving
print(times)

# i have to iterate thorugh times and append the values to dataframe . that can be done by iterating with the step of two
for i in range(0, len(times), 2):
    # i need dictionary to make a pair of start and end time.
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("times.csv")
video.release()
cv2.destroyAllWindows
