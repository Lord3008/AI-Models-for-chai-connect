import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { MessageSquare, Presentation, Users, TrendingUp, Send } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Link } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";

// TODO: Add your Gemini API key here or use environment variable
const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY || "";
const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";

const SoftSkills = () => {
  const { toast } = useToast();
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([
    { role: "assistant", content: "Hello! I'm your Soft Skills Coach. I can help you improve your communication, leadership, presentation skills, and professional etiquette. What would you like to work on today?" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    if (!GEMINI_API_KEY) {
      toast({
        title: "API Key Missing",
        description: "Please add your Gemini API key to use the chat feature.",
        variant: "destructive",
      });
      return;
    }

    const userMessage = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      // System prompt for soft skills coaching
      const systemPrompt = `You are an expert soft skills coach specializing in helping students and young professionals improve their:
- Communication skills (verbal and written)
- Presentation and public speaking
- Leadership and teamwork
- Professional etiquette and workplace behavior
- Emotional intelligence and interpersonal skills
- Time management and productivity
- Conflict resolution and negotiation

Provide practical, actionable advice with examples. Be encouraging, supportive, and specific. Keep responses concise but helpful. Answer in 5 small points always, don't elaborate much.`;

      const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: `${systemPrompt}\n\nUser: ${userMessage}`
            }]
          }]
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error("Gemini API error:", response.status, errorData);
        throw new Error(errorData.error?.message || `API error: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage = data.candidates?.[0]?.content?.parts?.[0]?.text || "I'm sorry, I couldn't generate a response.";
      
      setMessages(prev => [...prev, { role: "assistant", content: assistantMessage }]);
    } catch (error) {
      console.error("Error calling Gemini API:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to get response. Please check your API key and try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Header />
      <main className="pt-24 pb-16 bg-gradient-hero">
        <div className="container mx-auto px-4">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-12 animate-fade-in">
              <MessageSquare className="w-16 h-16 mx-auto mb-4 text-secondary" />
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                Soft <span className="bg-gradient-primary bg-clip-text text-transparent">Skills</span>
              </h1>
              <p className="text-xl text-muted-foreground">
                Develop essential communication, leadership, and interpersonal skills
              </p>
            </div>

            <div className="flex flex-col lg:flex-row gap-6">
              {/* Left Section - 70% */}
              <div className="lg:w-[55%] space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <MessageSquare className="w-8 h-8 mb-2 text-secondary" />
                      <CardTitle>Communication Skills</CardTitle>
                      <CardDescription>
                        Improve your verbal and written communication. Practice effective expression and active listening in a supportive environment.
                      </CardDescription>
                    </CardHeader>
                  </Card>

                  <Card>
                    <CardHeader>
                      <Presentation className="w-8 h-8 mb-2 text-secondary" />
                      <CardTitle>Presentation Skills</CardTitle>
                      <CardDescription>
                        Master the art of presenting ideas confidently. Get feedback on your presentations and practice public speaking.
                      </CardDescription>
                    </CardHeader>
                  </Card>

                  <Card>
                    <CardHeader>
                      <Users className="w-8 h-8 mb-2 text-secondary" />
                      <CardTitle>Leadership & Teamwork</CardTitle>
                      <CardDescription>
                        Develop leadership qualities and learn to work effectively in teams. Build skills through collaborative projects.
                      </CardDescription>
                    </CardHeader>
                  </Card>

                  <Card>
                    <CardHeader>
                      <TrendingUp className="w-8 h-8 mb-2 text-secondary" />
                      <CardTitle>Professional Etiquette</CardTitle>
                      <CardDescription>
                        Learn workplace etiquette, email writing, interview skills, and other professional soft skills from seniors.
                      </CardDescription>
                    </CardHeader>
                  </Card>
                </div>

                <Card className="bg-gradient-primary text-primary-foreground">
                  <CardHeader>
                    <CardTitle className="text-2xl">Skill Development Areas</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      {[
                        "Public Speaking",
                        "Negotiation",
                        "Time Management",
                        "Conflict Resolution",
                        "Emotional Intelligence",
                        "Critical Thinking",
                        "Networking",
                        "Problem Solving"
                      ].map((skill) => (
                        <div key={skill} className="bg-primary-foreground/10 rounded-lg p-3 text-center font-medium">
                          {skill}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Right Section - 30% - Chat Interface */}
              <div className="lg:w-[45%]">
                <Card className="sticky top-24 h-[calc(100vh-150px)] max-h-[600px] flex flex-col">
                  <CardHeader className="flex-none">
                    <CardTitle>AI Soft Skills Coach</CardTitle>
                    <CardDescription>Chat with AI to improve your soft skills</CardDescription>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col overflow-hidden p-0">
                    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 scrollbar-thin">
                      {messages.map((message, index) => (
                        <div
                          key={index}
                          className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} w-full`}
                        >
                          <div
                            className={`max-w-[85%] break-words rounded-lg px-4 py-2 ${
                              message.role === "user"
                                ? "bg-primary text-primary-foreground"
                                : "bg-muted"
                            }`}
                            style={{ 
                              wordBreak: 'break-word', 
                              overflowWrap: 'break-word',
                              whiteSpace: 'pre-wrap'
                            }}
                          >
                            <p className="text-sm">{message.content}</p>
                          </div>
                        </div>
                      ))}
                      {isLoading && (
                        <div className="flex justify-start">
                          <div className="bg-muted rounded-lg px-4 py-2">
                            <p className="text-sm">Typing...</p>
                          </div>
                        </div>
                      )}
                      <div ref={messagesEndRef} />
                    </div>
                    <form onSubmit={handleSubmit} className="flex-none p-4 border-t bg-background">
                      <div className="flex gap-2">
                        <Input
                          value={input}
                          onChange={(e) => setInput(e.target.value)}
                          placeholder="Ask about soft skills..."
                          disabled={isLoading}
                          className="flex-1"
                        />
                        <Button type="submit" size="icon" disabled={isLoading}>
                          <Send className="w-4 h-4" />
                        </Button>
                      </div>
                    </form>
                  </CardContent>
                </Card>
              </div>
            </div>

            <div className="text-center mt-8">
              <Link to="/features">
                <Button variant="outline" size="lg">
                  Back to Features
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default SoftSkills;
