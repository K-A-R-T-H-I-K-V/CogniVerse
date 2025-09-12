"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Brain, Book, Send, RotateCcw } from "lucide-react"
import ReactMarkdown from "react-markdown"
import rehypeRaw from "rehype-raw"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

interface Message {
  id: string
  type: "user" | "ai"
  content: string
  timestamp: Date
  imageUrl?: string
}

interface Source {
  type: "text" | "image"
  content: string
  page?: number
  filename?: string
}

interface ApiResponse {
  answer: string
  sources: Source[]
  relevant_image?: string
}

export default function CogniVerse() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [sources, setSources] = useState<Source[]>([])
  const [leftWidth, setLeftWidth] = useState(70)
  const [isDragging, setIsDragging] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleMouseDown = () => {
    setIsDragging(true)
  }

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return

    const containerWidth = window.innerWidth
    const newLeftWidth = (e.clientX / containerWidth) * 100

    // Constrain between 30% and 80%
    if (newLeftWidth >= 30 && newLeftWidth <= 80) {
      setLeftWidth(newLeftWidth)
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = "col-resize"
      document.body.style.userSelect = "none"
    } else {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
  }, [isDragging])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input.trim(),
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: input.trim() }),
      })

      if (!response.ok) {
        throw new Error("Failed to get response from AI")
      }

      const data: ApiResponse = await response.json()

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: data.answer,
        timestamp: new Date(),
        imageUrl: data.relevant_image,
      }

      setMessages((prev) => [...prev, aiMessage])
      setSources(data.sources)
    } catch (error) {
      console.error("Error:", error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: "Sorry, I encountered an error while processing your question. Please try again.",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const resetChat = async () => {
    try {
      await fetch("http://127.0.0.1:5000/reset", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      })
    } catch (error) {
      console.error("Error resetting chat:", error)
    }

    setMessages([])
    setSources([])
    setInput("")
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex h-screen text-foreground" style={{ backgroundColor: "#1E1C3A" }}>
      {/* Left Column - Chat Interface */}
      <div className="flex flex-col border-r border-border" style={{ width: `${leftWidth}%` }}>
        {/* Header */}
        <div
          className="flex items-center justify-between p-4 border-b border-border"
          style={{ backgroundColor: "#2D3748" }}
        >
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-accent animate-pulse">
              <Brain className="w-6 h-6 drop-shadow-[0_0_8px_#00BFFF]" />
              <Book className="w-6 h-6 drop-shadow-[0_0_8px_#00BFFF]" />
            </div>
            <h1 className="text-2xl font-bold text-foreground font-inter">CogniVerse</h1>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={resetChat}
            className="flex items-center gap-2 bg-transparent border-accent/50 text-accent hover:border-accent hover:shadow-[0_0_20px_#00BFFF] hover:text-white transition-all duration-300 hover:scale-105"
          >
            <RotateCcw className="w-4 h-4" />
            Reset Chat
          </Button>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-muted-foreground">
                <div className="flex items-center justify-center gap-2 mb-4 text-accent">
                  <Brain className="w-8 h-8 drop-shadow-[0_0_12px_#00BFFF] animate-pulse" />
                  <Book className="w-8 h-8 drop-shadow-[0_0_12px_#00BFFF] animate-pulse" />
                </div>
                <h2 className="text-xl font-semibold mb-2 text-foreground font-inter">Welcome to CogniVerse!</h2>
                <p className="font-inter">
                  Your personal AI study buddy. Ask me a question about your textbook to get started.
                </p>
              </div>
            </div>
          ) : (
            <>
                {messages.map((message) => (
                <div key={message.id}>
                  {/* --- Renders the Text Bubble (only if there is text) --- */}
                  {message.content && (
                    <div className={`flex w-full ${message.type === "user" ? "justify-end" : "justify-start"}`}>
                      <div
                        className={`max-w-[80%] rounded-lg p-4 transition-all duration-300 font-inter ${
                          message.type === "user"
                            ? "text-card-foreground hover:shadow-[0_0_15px_rgba(0,191,255,0.3)] hover:scale-[1.02]"
                            : "text-card-foreground border-l-4 border-accent hover:shadow-[0_0_15px_rgba(0,191,255,0.3)] hover:border-l-8"
                        }`}
                        style={{ backgroundColor: "#2D3748" }}
                      >
                        {message.type === "ai" ? (
                          <div className="prose prose-invert max-w-none text-white">
                            <ReactMarkdown>{message.content}</ReactMarkdown>
                          </div>
                        ) : (
                          <p>{message.content}</p>
                        )}
                      </div>
                    </div>
                  )}

                  {/* --- Renders the Clickable Image Bubble (only for AI messages with an image) --- */}
                  {message.type === "ai" && message.imageUrl && (
                    <div className="flex w-full justify-start mt-2">
                      <Dialog>
                        <DialogTrigger asChild>
                          <div className="max-w-[80%] rounded-lg p-2 cursor-pointer transition-transform duration-200 hover:scale-[1.03] hover:shadow-[0_0_25px_rgba(0,191,255,0.4)]" style={{ backgroundColor: "#2D3748" }}>
                            <img
                              src={message.imageUrl}
                              alt="Relevant Diagram (Click to enlarge)"
                              className="w-full max-w-lg rounded-md object-contain"
                            />
                          </div>
                        </DialogTrigger>
                        {/* <DialogContent className="sm:max-w-[80vw] max-w-[95vw] p-0 border-none bg-transparent"> */}
                        <DialogContent className="sm:max-w-[80vw] max-w-[95vw] bg-[#2D3748] border-accent/50">
                          <img
                            src={message.imageUrl}
                            alt="Enlarged Diagram"
                            className="w-full h-full object-contain max-h-[90vh] mx-auto"
                          />
                        </DialogContent>
                      </Dialog>
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div
                    className="border-l-4 border-accent rounded-lg p-4 max-w-[80%] shadow-[0_0_20px_rgba(0,191,255,0.4)] animate-pulse font-inter"
                    style={{ backgroundColor: "#2D3748", color: "#f8fafc" }}
                  >
                    <div className="flex items-center gap-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-accent rounded-full animate-bounce shadow-[0_0_8px_#00BFFF]"></div>
                        <div
                          className="w-2 h-2 bg-accent rounded-full animate-bounce shadow-[0_0_8px_#00BFFF]"
                          style={{ animationDelay: "0.1s" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-accent rounded-full animate-bounce shadow-[0_0_8px_#00BFFF]"
                          style={{ animationDelay: "0.2s" }}
                        ></div>
                      </div>
                      <span className="text-muted-foreground">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Chat Input */}
        <div className="p-4 border-t border-border" style={{ backgroundColor: "#2D3748" }}>
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your textbook..."
              className="flex-1 min-h-[60px] max-h-[120px] resize-none border-border text-foreground placeholder:text-muted-foreground focus:border-accent focus:shadow-[0_0_20px_rgba(0,191,255,0.3)] focus:ring-2 focus:ring-accent/50 transition-all duration-300 font-inter"
              style={{ backgroundColor: "#1f2937" }}
              disabled={isLoading}
            />
            <Button
              type="submit"
              size="lg"
              disabled={!input.trim() || isLoading}
              className="px-4 bg-accent hover:bg-accent/90 text-accent-foreground hover:shadow-[0_0_30px_#00BFFF,0_0_60px_#00BFFF,0_0_90px_#00BFFF] hover:scale-110 transition-all duration-300 hover:animate-pulse disabled:hover:shadow-none disabled:hover:scale-100 disabled:hover:animate-none"
            >
              <Send className="w-5 h-5" />
            </Button>
          </form>
        </div>
      </div>

      {/* Draggable Resizer */}
      <div
        className="w-1 bg-border hover:bg-accent cursor-col-resize transition-colors duration-200 hover:shadow-[0_0_10px_#00BFFF] relative group"
        onMouseDown={handleMouseDown}
      >
        <div className="absolute inset-y-0 -left-1 -right-1 group-hover:bg-accent/20 transition-colors duration-200" />
      </div>

      {/* Right Column - Source Viewer */}
      <div className="flex flex-col" style={{ width: `${100 - leftWidth}%`, backgroundColor: "#1f2937" }}>
        {/* Header */}
        <div className="p-4 border-b border-sidebar-border" style={{ backgroundColor: "#2D3748" }}>
          <h2 className="text-lg font-semibold text-sidebar-foreground font-inter">Retrieved Context</h2>
        </div>

        {/* Sources */}
        <div className="flex-1 overflow-y-auto p-4">
          {sources.length === 0 ? (
            <div className="flex items-center justify-center h-full text-center text-muted-foreground">
              <p className="font-inter">The sources used for the AI's answer will appear here.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {sources.map((source, index) => (
                <Card
                  key={index}
                  className="border-sidebar-border hover:shadow-[0_0_15px_rgba(0,191,255,0.2)] hover:border-accent/50 transition-all duration-300 hover:scale-[1.02] cursor-pointer"
                  style={{ backgroundColor: "#374151" }}
                >
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm text-sidebar-primary-foreground font-inter">
                      {source.type === "text" ? `Source: Page ${source.page}` : `Image: ${source.filename}`}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {source.type === "text" ? (
                      <p className="text-sm text-sidebar-accent-foreground leading-relaxed font-inter">
                        {source.content}
                      </p>
                    ) : (
                      <img
                        src={source.content || "/placeholder.svg"}
                        alt={source.filename}
                        className="w-full rounded-md hover:shadow-[0_0_20px_rgba(0,191,255,0.3)] transition-all duration-300"
                      />
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
