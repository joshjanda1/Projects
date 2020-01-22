library(shiny)
library(tidyverse)
library(reticulate)
library(shinyWidgets)

##PLEASE ALTER AND RUN TO AVOID ERROR DUE TO PLOTTING IN PYTHON WITH RETICULATE##
py_run_string("import os as os")
py_run_string("os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Users/Josh/Anaconda3/Library/plugins/platforms'")
#py_run_string("os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Path_to_your_anaconda/anaconda3/Library/plugins/platforms'")
##
library(keras)
neural_net = load_model_hdf5('conv_nn.h5')
neural_net %>% compile(optimizer = 'adam',
                       loss = 'categorical_crossentropy',
                       metrics = 'accuracy')

reticulate::source_python("shiny_backend.py")

names = 0:9

model_architecture = data.frame("Layer Number" = c(1:17),
                                "Layer Type" = c("Conv2D", "MaxPool2D", "BatchNormalization",
                                                 "Conv2D", "MaxPool2D", "BatchNormalization",
                                                 "Conv2D", "MaxPool2D", "BatchNormalization",
                                                 "Dropout", "Flatten", "FC", "Dropout",
                                                 "BatchNormalization", "FC", "BatchNormalization",
                                                 "FC"),
                                "Filters" = c(128, 128, 128, 196, 196, 196, 256, 256, 256, 256, rep("-", 7)),
                                "Size" = c("26x26", rep("13x13", 2), "11x11", rep("5x5", 2), "3x3", rep("1x1", 3), rep("-", 7)),
                                "Kernel Size" = c("3x3", "2x2", "-",
                                                  "3x3", "2x2", "-",
                                                  "3x3", "2x2", "-",
                                                  rep("-", 8)),
                                "Stride" = c("1", "1", "-",
                                             "1", "1", "-",
                                             "1", "1", "-",
                                             rep("-", 8)),
                                "Activation" = c(rep("relu", 16), "softmax"))
ui = fluidPage(
  
  title = "Predicting Users Handwritten Digits",
  titlePanel("Click Plot to Begin/Stop Drawing"),
  fluidRow(
    
    sidebarLayout(position = "right",
      
      sidebarPanel(h4("Project Motivation"),
                   p("This Shiny Application was created to allow users to draw their own handwritten digits (0-9), based on the Mnist dataset."),
                   p("The model creating these predictictions is a Convolutional Neural Network"),
                   p("The architecture of this model is:"),
                   tableOutput("model_architecture"), width = 4),
      
      mainPanel(
        
        HTML("<div style='overflow:auto'>"),
        
        plotOutput("drawingboard", width = "600px", height = "800px",
                           hover = hoverOpts(id = "hover", delay = 100, delayType = "throttle", clip = TRUE,
                                             nullOutside = TRUE),
                           click = "click"),
        HTML("</div>"),
        width = 8)
    )
  ),
  
  hr(),
  
  fluidRow(
    
    column(3,
           
           h4("Predicting Handwritten Digits - CNN Trained on Mnist Data"),
           sliderInput("pencilwidth", "Width of Stylus",
                       min = 1, max = 30, step = 1, value = 10),
           
           br(),
           
           actionButton("resetplot", "Reset Plot"),
           actionButton("predict", "Predict Drawn Number"),
           
           br(),
           br(),
           
           textOutput('predictednumber'),
  
           br(),
           
           tags$a(href = "https://github.com/joshjanda1",
                  tags$img(src = 'github.png', title = "Created By Josh Janda")),
           
           br(),
           br(),
           
           tags$p("Created By Josh Janda")
    ),
    
    column(4, offset = 1,
           plotOutput('probplot'))
  )
)


server = function(input, output, session) {
  
  vals = reactiveValues(x = NULL, y = NULL) #x and y are changing with user input so reactive
  draw = reactiveVal(FALSE) #drawing board (pixel input)
  pixels = reactiveValues(fig = NULL, mtx = NULL)
  predictions = reactiveValues(classes = "NA", probs = NULL)
  observeEvent(input$click, handlerExpr = {
    
    temp = draw()
    draw(!temp) #on click enable draw
    
    if (!draw()) {
      
      vals$x = c(vals$x, NA) #if draw not true then set x and y to a vector of itself and NA to stop drawing
      vals$y = c(vals$y, NA)
      
    }
  })
  
  observeEvent(input$resetplot, handlerExpr = {
    
    vals$x = NULL
    vals$y = NULL #sets x and y to null when resetting plot to make plot blank
    pixels$prediction = NULL
    pixels$fig = NULL
    pixels$mtx = NULL
    output$probsplot = NULL
    
  })
  
  
  observeEvent(input$hover, {
    
    if (draw()) {
      
      vals$x = c(vals$x, input$hover$x) # if draw enabled, wherever user hovers draw that x value and combine with current drawn
      vals$y = c(vals$y, input$hover$y) # if draw enabled, wherever user hovers draw that y value and combine with current drawn
      
    }
    
  })
  
  observeEvent(input$predict, {
    
    pixels$fig = make_fig(vals$x, vals$y)
    pixels$mtx = get_pixels(pixels$fig)
    
    dim(pixels$mtx) = c(1, 28, 28, 1)
    
    #predictions$classes = predict_pixels(pixels$mtx)
    #predictions$probs = pixel_probs(pixels$mtx)
    predictions$classes = neural_net %>% predict_classes(pixels$mtx) #### these compute the prediction when predict button clicked
    predictions$probs = neural_net %>% predict_proba(pixels$mtx)
    
    output$probplot = renderPlot({
      
      barplot(predictions$probs, main = "Neural Network Probability Outputs", horiz = TRUE,
              xlab = "Probability", col = rainbow(9), names.arg = names) # plot probability outputs from neural network
      
    })
    
  })
  
  output$drawingboard = renderPlot({

    plot(x = vals$x, y = vals$y, xlim = c(0, 28), ylim = c(0, 28),
         xlab = "x-pixels", ylab = "y-pixels", type = "l", lwd = input$pencilwidth) # plot drawing
    
  })
  
  output$predictednumber = renderText({
    
    paste("Predicted Value: ", predictions$classes) # display predicted value
    
  })
  
  output$model_architecture = renderTable(model_architecture)
  
}

shinyApp(ui, server)
