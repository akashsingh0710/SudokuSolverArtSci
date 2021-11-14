import cv2
import glob
import os
import numpy as np
from PIL import Image as im
import pytesseract   


#### FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours , img):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area>max_area and area> 0.8*len(img)*len(img[0]):
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015*peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

#### ---- CHANGE THE BELOW PATH AS PER THE CURRENT DIRECTORY ---------
Sudoku_digits = "C:\\Users\\Akash MSI\Desktop\\SudokuSolverArtSci\\digits\\" 

#### CHARACTER RECOGNITION
def text_it(arr , x , y):
    #config = ('-l eng --oem 1 --psm 6 ')
    # config = ('-l eng --oem 2 --psm 6 ')
    #config = ('-l eng --oem 2 --psm 6 ')
    # config = r'--oem 3 --psm 6 outputbase digits'
    
    #psm working well 6 - problem detecting 9
    #psm working well 7 - problem detecting 5 and 9
    #psm working well 8, 10 - problem detecting some 5 and 9
    
    
    config1 = r'--oem 0 --psm 6 -c tessedit_char_whitelist=123456789'
    config2 = r'--oem 1 --psm 6 -c tessedit_char_whitelist=123456789'
    
    arr = np.uint8(arr)
    im_pil = im.fromarray(arr)
    
    a,b = arr.shape
    text = ''
    if a and b:
        size =  b * 2 , a * 2
    
        im_resized = im_pil.resize(size, im.ANTIALIAS)
        
        cv2.imwrite(Sudoku_digits + "Sudoku_digits_{}_{}.jpg".format(x+1, y+1) , arr)
        
        
        text1 = pytesseract.image_to_string(im_resized , lang='eng', config = config1)
        print("Processing...")
        
    return text1


#### HELPER FUNCTION TO PRINT DETECTED SUDOKU GRID TO THE CONSOLE
def display_board(grid_patt):   
    for i in range(1,10):
    
        for j in range(1,10):
            print(int(grid_patt[i-1][j-1]), end=" ")
        # print(f'j vale was: {j}')
            if j==9:
                print('\n' , end="")
            elif (j)%3 == 0:
                print('|', end=" ")
        if (i)%3 == 0 and i!=9:
            print('- - -   - - -   - - - ')   
            


#### ---- SOLVING SUDOKU USING BACKTRACKING ---------
def is_empty(i,j,grid_patt):
  if grid_patt[i-1][j-1] <= 0:
      return 1
  else: 
      return 0
 
    
def in_row(i,j,grid_patt,n):
    for k in range(1,10):
        if grid_patt[i-1][k-1] == n:
            return 1
    return 0  

        
def in_column(i,j,grid_patt,n):
    for k in range(1,10):
        if grid_patt[k-1][j-1] == n:
            return 1
    return 0  


def in_block(i,j,grid_patt,n):
    for k in range(int((i-1)/3)*3+1,int((i-1)/3)*3+4):
        for l in range(int((j-1)/3)*3+1,int((j-1)/3)*3+4):
            if grid_patt[k-1][l-1] == n:
                return 1
    return 0     
        
           
def empty_cell(grid_patt):
    temp=[]
    for i in range(1,10):
        for j in range(1,10):
            if grid_patt[i-1][j-1] <= 0:
                k = [i,j]
                
                temp.append(k)            
    return temp    

 

def backtrack_solver(empty_cell_pos , grid_patt , i):
    
    if i>= len(empty_cell_pos):
         print('\n')
         print('\n')
         print('Solved Board:')
         display_board(grid_patt)
         return grid_patt
            
    a,b= empty_cell_pos[i]
    for digit in range(1,10):
        if (in_row(a,b,grid_patt,digit) ==0 and in_column(a,b,grid_patt,digit)==0 and in_block(a,b,grid_patt,digit) ==0):
            if digit!=None:
                grid_patt[a-1][b-1] = digit
            
            backtrack_solver(empty_cell_pos , grid_patt , i+1)
    grid_patt[a-1][b-1] = 0




#### ---- CHANGE THE BELOW PATH AS PER THE CURRENT DIRECTORY ---------
Sudoku_Image = "C:\\Users\\Akash MSI\Desktop\\SudokuSolverArtSci\\ProblemInput"
total_input_files = len(os.listdir(Sudoku_Image))

for j in range (0,total_input_files):
    images = [cv2.imread(file) for file in glob.glob(Sudoku_Image + "\\*.jpg" .format(j+1)) ]
    
    
for i in range (0 , len(images)):
    image_no = i+1
    img_orig = images[i]  

    img = im.fromarray(img_orig)
    a,b = img.size

    if b>=a:
        size =  720 , int((b/a)*720)  
    else:
        size =   int((a/b)*720)   ,720  
    
    img = img.resize(size, im.ANTIALIAS)
    
    
    resized = np.array(img) 
    
    cv2.imshow("resized", resized)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("resized_and_gray.jpg", gray)

    
    params = [ (11, 41, 21)]
    for (diameter, sigmaColor, sigmaSpace) in params:
        # cv2.imshow('Original image',gray)
        blurred = cv2.bilateralFilter(gray, diameter, sigmaColor, sigmaSpace)
        
    cv2.imwrite("blurred.jpg", blurred)    
        
    process = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite("threshold.jpg", process)   
    
     
 #----------------------Dialating-------------------------------   
    
    process = cv2.bitwise_not(process, process)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    process = cv2.dilate(process, kernel,iterations = 1)
    cv2.imwrite("dilated.jpg", process) 
    
    #DIALATED IMAGE
    #cv2.imshow("Dialated", process)
       
    connectivity = 4
    output1= cv2.connectedComponentsWithStats(process, connectivity , cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output1

    max_area = 0
    ind = 0
    for j in range (0,stats.shape[0]):
        x = stats[j, cv2.CC_STAT_LEFT]
        y = stats[j, cv2.CC_STAT_TOP]
        w = stats[j, cv2.CC_STAT_WIDTH]
        h = stats[j, cv2.CC_STAT_HEIGHT]
        
        area = w*h
        if area > max_area and area < (len(process)-2)*(len(process[0]-2)) and x!=0 and y!=0:
            max_area = area
            ind = j


    x = stats[ind, cv2.CC_STAT_LEFT]
    y = stats[ind, cv2.CC_STAT_TOP]
    w = stats[ind, cv2.CC_STAT_WIDTH]
    h = stats[ind, cv2.CC_STAT_HEIGHT]
    (cX, cY) = centroids[ind]
    
    output = process.copy()
    
    cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 13)
    cv2.circle(resized, (int(cX), int(cY)), 4, (0, 0, 255), -1)
       
    #####################RECOGNIZED SUDOKU IMAGE
    cv2.imshow("Output", resized)
    cv2.imwrite("sudoku_ROI.jpg", resized) 
    
    top_r, top_l, bottom_l, bottom_r = (x+w, y), (x, y), (x , y + h), (x + w, y + h)
    
    roi = output[y:y+h , x:x+w ]
    roi_org = gray[y:y+h , x:x+w ]
    
    image_sudoku_gray = roi_org.copy()
    image_sudoku_bw = roi_org.copy() 
      
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #FIND ALL CONTOURS
    cv2.drawContours(roi_org, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS
    
    ####################   FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
    biggest, maxArea = biggestContour(contours , roi) # FIND THE BIGGEST CONTOUR
      
    roi_org2 = roi_org.copy()
    
    if biggest.size != 0:
        biggest = reorder(biggest)
        flag=1
        cv2.drawContours(roi_org, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
        
        ################  SUDOKU CORNER IMAGE
        cv2.imshow('Corners Detected',roi_org)
        cv2.imwrite("corners.jpg", roi_org)
        ################ Extracting the points
        corners = [(corner[0][0], corner[0][1]) for corner in biggest]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        

        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
    
        width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
        width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
        width = max(int(width_A), int(width_B))


        height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
        height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
        height = max(int(height_A), int(height_B))

    
        input_pts = np.float32([top_r, bottom_l , bottom_r , top_l])
        output_pts = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])




        grid = cv2.getPerspectiveTransform(input_pts, output_pts)
        image_sudoku_bw = cv2.warpPerspective(roi, grid, (width, height))
        image_sudoku_gray = cv2.warpPerspective(roi_org2, grid, (width, height))
        
            
    cv2.imshow('image_sudoku_bw',image_sudoku_bw)
    cv2.imshow('image_sudoku_gray',image_sudoku_gray)
    cv2.imwrite("perspectivetransformation.jpg", image_sudoku_gray)
      
    text_pop = np.zeros((9,9))
    
    
    a,b = image_sudoku_bw.shape
    
    for x in range (0, 9,1):
        for y in range (0, 9,1):
            
            
            sa =image_sudoku_gray[x*int(a/9)+10 : (x+1)*int(a/9)-10 , y*int(b/9)+10 : (y+1)*int(b/9)-10]    
            sa2 = cv2.threshold(sa, 140, 255, cv2.THRESH_BINARY)[1]
            text_out = text_it(np.array(sa2) , x , y)
            numeric_filter = filter(str.isdigit, text_out)
            numeric_string = "".join(numeric_filter)
            if numeric_string.isdigit():
                if int(numeric_string) >0 and int(numeric_string) <10:
                    text_pop[x,y] = int(numeric_string)    
      
        
    sudoku_que = text_pop.tolist()  
    input_grid = sudoku_que
    
    print('')
    print('')
    print('Original Board:')
    display_board(input_grid)    
    
    empty_cell_pos = empty_cell(input_grid)
    solved_grid = backtrack_solver(empty_cell_pos , input_grid , 0)

    print('')
    print('')    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
    