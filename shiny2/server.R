function(input, output, session) {
  # Define a reactive expression for the document term matrix
  terms <- reactive({
    # Change when the "update" button is pressed...
    input$update
    # ...but not for anything else
    isolate({
      withProgress({
        setProgress(message = "Processing corpus...")
        getTermMatrix(input$selection)
      })
    })
  })

  # Make the wordcloud drawing predictable during a session
  wordcloud_rep <- repeatable(wordcloud)

  output$plot <- renderPlot({
    v <- terms()
    wordcloud_rep(names(v), v, scale=c(4,0.5),
                  min.freq = input$freq, max.words=input$max,
                  colors=brewer.pal(8, "Dark2"))

  output$myImage <- renderImage({
  # A temp file to save the output.
  # This file will be removed later by renderImage
    outfile <- tempfile(fileext='/Users/shaokuixing/Desktop/Python/shiny2/image.png')

  # Generate the PNG
    png(outfile, width=400, height=300)
    hist(rnorm(input$obs), main="Generated in renderImage()")
    dev.off()

  # Return a list containing the filename
    list(src = outfile,
    contentType = 'image/png',
                       width = 400,
                       height = 300,
                       alt = "This is alternate text")
                }, deleteFile = TRUE)
  })
}
