library(shiny)
library(tidyverse)
library(reticulate)
library(shinyWidgets)

##PLEASE ALTER AND RUN TO AVOID ERROR DUE TO PLOTTING IN PYTHON WITH RETICULATE##
py_run_string("import os as os")
py_run_string("os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Users/Josh/Anaconda3/Library/plugins/platforms'")
#py_run_string("os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Path_to_your_anaconda/anaconda3/Library/plugins/platforms'")
##


reticulate::source_python("shiny_backend.py")
names = 0:9

ui = fluidPage(

  setBackgroundColor("#A8A8A8"),
  
  h1(id = "title", "Predicting Hand-Written Digits Using a Convolutional Neural Network"),
  h4("Click on plot to start/stop drawing"),
  fluidRow(
    sidebarLayout(
      sidebarPanel(
        
        sliderInput("pencilwidth", "Width of the Pencil", min = 1, max = 30, step = 1, value = 5),
        actionButton("resetplot", "Reset Plot"),
        actionButton("predict", "Predict Number"),
        textOutput("selected"),
        plotOutput("probplot"),
        
        tags$a(href = "https://github.com/joshjanda1",
               tags$img(src = 'github.png', title = "Created By Josh Janda"))),
      
      
      mainPanel(
        
        plotOutput("drawingboard", width = "500px", height = "500px",
                   hover = hoverOpts(id = "hover", delay = 100, delayType = "throttle", clip = TRUE,
                                     nullOutside = TRUE),
                   click = "click")
      )
    )
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
    predictions$classes = predict_pixels(pixels$mtx) #### these compute the prediction when predict button clicked
    predictions$probs = pixel_probs(pixels$mtx)
    
    output$probplot = renderPlot({
      
      barplot(predictions$probs, main = "Neural Network Probability Outputs", horiz = TRUE,
              xlab = "Probability", col = rainbow(9), names.arg = names) # plot probability outputs from neural network
      
    })
    
  })
  
  output$drawingboard = renderPlot({

    plot(x = vals$x, y = vals$y, xlim = c(0, 28), ylim = c(0, 28),
         xlab = "x-pixels", ylab = "y-pixels", type = "l", lwd = input$pencilwidth) # plot drawing
    
  })
  
  output$selected = renderText({
    
    paste("Predicted Value: ", predictions$classes) # display predicted value
    
  })
  
}

shinyApp(ui, server)
