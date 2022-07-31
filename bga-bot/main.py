import os
import time

import cv2
from PIL import ImageGrab, Image
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' # windows tessaract fix

BROWSER_BOX = (0, 0, 937, 1070)
FIRST_NUM_BOX = (16, 384, 30, 415)
SECOND_NUM_OFFSET = 14
NEXT_CARD_OFFSET = 111

num_players = 3

def get_board_cards(img):

    def img_to_card(img):
        # card_img = process_img(img)
        save_img(img)
        card = pytesseract.image_to_string(img, config='--psm 10')
        print(card)
        return card if len(card) == 1 else card[0]

    cards = []
    for i in range(num_players):
        x1, y1, x2, y2 = FIRST_NUM_BOX
        x1 += NEXT_CARD_OFFSET * i
        x2 += NEXT_CARD_OFFSET * i
        first_box = (x1, y1, x2, y2)
        second_box = (x1 + SECOND_NUM_OFFSET, y1, x2 + SECOND_NUM_OFFSET, y2)

        card = img_to_card(img.crop(first_box)) + img_to_card(img.crop(second_box))
        cards.append(card)

    return cards

def process_img(img):
    # convert img to opencv for adaptive thresholding
    opencv_img = np.array(img)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    opencv_img = cv2.adaptiveThreshold(opencv_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)

    return opencv_img

def save_img(opencv_img):
    # convert opencv img back to PIL and save
    time_str = str(time.time())
    # opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    # img = Image.fromarray(opencv_img)
    opencv_img.save(os.getcwd() + f'\\img\\{time_str}.png', 'PNG')

def main():
    game_img = ImageGrab.grab(BROWSER_BOX)
    cards = get_board_cards(game_img)

    print(cards)

if __name__ == '__main__':
    main()
