import cv2 as cv
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import tensorflow as tf
from skimage.transform import resize

board = []


def get_num(cell):
    """
    get specific cell from the board, check whether number exists in the cell and if so returns
    numpy array describing the digit

    :param cell: cell from sudoku board
    :return: numpy array describing the digit
    """

    # assign each pixel that is not black to white
    ret, thresh = cv.threshold(cell, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    thresh = cv.resize(thresh, (300, 300), interpolation=cv.INTER_CUBIC)
    # clean pixels of the border's cell
    thresh1 = clear_border(thresh, buffer_size=25)
    # find all the features(contours) in the cell
    _, contours_digit, hierarchies_digit = cv.findContours(thresh1.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours_digits = sorted(contours_digit, key=lambda x: cv.contourArea(x), reverse=True)

    if len(sorted_contours_digits) > 0:
        blank = np.zeros(thresh1.shape, dtype='uint8')
        # draw the biggest contour above black pixels
        cv.drawContours(blank, [sorted_contours_digits[0]], -1, 255, -1)
        # Calculate the relative share of the biggest contour (by area) we found above
        not_zeros = cv.countNonZero(blank) / float(thresh1.shape[0]*thresh1.shape[1])

        if not_zeros > 0.04:
            # so there is a number in the cell
            x_corner, y_corner, w, h = cv.boundingRect(sorted_contours_digits[0])
            digit = thresh[y_corner:y_corner + h, x_corner:x_corner + w]

            left, right, upper, lower = find_bounderies(digit)
            width = right - left + 1
            height = lower - upper + 1

            square_number = create_square_around_digit(digit, height, width, upper, left)
            padded_number = paddig_with_black_pixels(square_number)

            #cv.imshow("padded_number", padded_number)
            #cv.waitKey(0)
            return padded_number
    return None


def find_bounderies(contour_pixels):
    """
    find the borders of the digit in the cell
    :param contour_pixels: numpy array with the pixels of the digit only
    :return: the bounderies of the digit - left, right, upper, lower
    """

    left = contour_pixels.shape[0]
    right = 0
    upper = contour_pixels.shape[1]
    lower = 0
    for row in range(contour_pixels.shape[0]):
        for col in range(contour_pixels.shape[1]):
            if contour_pixels[row][col] > 20:
                if row < upper:
                    upper = row
                elif row > lower:
                    lower = row
                if col < left:
                    left = col
                elif col > right:
                    right = col
    return left, right, upper, lower


def create_square_around_digit(contour_pixels, height, width, upper, left):
    """
    :param contour_pixels: numpy array with the pixels of the digit only
    :param height: the height boundery
    :param width: the width boundery
    :param upper: the upper boundery
    :param left: the left boundery
    :return: the digit bounded by a rectangle
    """

    number_pixels = np.zeros((height, width))

    # insert digit's pixels into 'number_pixel'
    for row in range(height):
        for col in range(width):
            number_pixels[row][col] = contour_pixels[upper + row][left + col]

    if height != width:
        rect_num = np.zeros((max((height, width)), max(height, width)))
        if height > width:
            # pad black pixels to left and right
            remain = int((height - width) / 2)
            rect_num[:, remain:remain + width] = number_pixels
        else:
            # pad black pixels beneath and above
            remain = int((width - height) / 2)
            rect_num[remain:remain + height, :] = number_pixels
    else:
        rect_num = number_pixels
    return rect_num


def paddig_with_black_pixels(rect_number):
    """
    pad the digit with 4-black_pixels at least in order to prepare the numpy array for prediction
    :param rect_number: the digit bounded by a rectangle
    :return: numpy array with the pixels of the digit padded with black pixels
    """

    padded_num = np.zeros((28, 28))
    if np.any(np.array(rect_number.shape) > 20):
        padded_num[4:24, 4:24] = resize(rect_number, (20, 20))
    else:
        size = rect_number.shape
        reamin_height = int((28 - size[0]) / 2)
        reamin_width = int((28 - size[1]) / 2)
        padded_num[reamin_height:size[0] + reamin_height, reamin_width:size[1] + reamin_width] = rect_number
    return padded_num


def process_image(path):
    """
    process sudoku board image in order to predict the digit inside it
    :param path: the path to the sudoku board image
    :return: board with the value of the digits in the board image
    """

    img = cv.imread(path)
    # turn the image to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9)
    # find all the features(contours) in the image
    _, contours, hierarchies = cv.findContours(adaptive_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # sort the contours above by their area
    sorted_contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    contour_of_square = None

    for i in range(len(sorted_contours)):
        # epsilon is the maximum distance from contour to approximated the actual contour
        epsilon = 0.05 * cv.arcLength(sorted_contours[i], True)
        # approximates a contour shape to another shape with less number of vertices
        approx = cv.approxPolyDP(sorted_contours[i], epsilon, True)
        if len(approx) == 4:
            contour_of_square = approx
            break
    # deskews the square grid make it easier to determine cells
    clean_board = four_point_transform(adaptive_thresh, contour_of_square.reshape(4, 2))
    clean_board = cv.resize(clean_board, (300, 300), interpolation=cv.INTER_CUBIC)
    #cv.imshow("clean_board", clean_board)
    #cv.waitKey(0)
    return process_cells(clean_board)


def process_cells(clean_board):
    """
    loop over the board cells and process each one of them, than predict their value
    :param clean_board: the processed sudoku board after applying 'process_image' on the original image
    :return: board with the value of the digits in the board image
    """

    size_to_add_x = clean_board.shape[1] // 9
    size_to_add_y = clean_board.shape[0] // 9

    # load the model for digits prediction
    model = tf.keras.models.load_model('output/digit_classifier.h5', compile=False)

    for i in range(0, 9):
        row = []
        for j in range(0, 9):
            number = get_num(clean_board[(i * size_to_add_y):((i + 1) * size_to_add_y),
                             (j * size_to_add_x):((j + 1) * size_to_add_x)])
            if number is not None:
                # normalize the pixels
                number = number / 255
                number = np.reshape(number, newshape=(1, 28, 28, 1))
                prediction = model.predict(number).argmax(axis=1)[0]
                row.append(prediction)
            else:
                row.append(0)
        board.append(row)

    return board


def get_board(path):
    process_image(path)
    return board

