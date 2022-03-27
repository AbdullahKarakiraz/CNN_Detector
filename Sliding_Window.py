def sliding_window(image,step,ws):
    """
    image : input image
    step : number of steps of the window
    ws : windows size
    """
    for y in range(0,image.shape[0]-ws[1],step):
        for x in range(0,image.shape[1]-ws[0],step):
            
            yield(x,y,image[y:y+ws[1], x:x+ws[0]])           