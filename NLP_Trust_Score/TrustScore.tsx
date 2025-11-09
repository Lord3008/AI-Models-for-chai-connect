import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Award, Star, TrendingUp, Shield } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Link } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";

// TODO: Add your Gemini API key here or use environment variable
const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY || "";
const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";

const TrustScore = () => {
  const { toast } = useToast();
  const [reviews, setReviews] = useState("");
  const [trustScore, setTrustScore] = useState<number | null>(null);
  const [analysis, setAnalysis] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleCalculateScore = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!reviews.trim()) {
      toast({
        title: "No Reviews Provided",
        description: "Please enter some reviews to calculate trust score.",
        variant: "destructive",
      });
      return;
    }

    if (!GEMINI_API_KEY) {
      toast({
        title: "API Key Missing",
        description: "Please add your Gemini API key in environment variables (VITE_GEMINI_API_KEY).",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);

    try {
      const systemPrompt = `You are a trust score analyzer for a peer-to-peer learning platform called bytee. 
Analyze the provided reviews and feedback about a user and calculate a trust score out of 5.

Consider the following factors:
- Quality and consistency of help provided
- Positive feedback from peers
- Reliability and punctuality
- Communication skills
- Expertise demonstrated
- Number of successful exchanges
- Negative feedback or complaints

Return ONLY a JSON object with this exact structure (no additional text):
{
  "score": <number between 0-5 with one decimal>,
  "analysis": "<brief 2-3 sentence explanation of the score>"
}`;

      const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: `${systemPrompt}\n\nReviews to analyze:\n${reviews}`
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
      const responseText = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
      
      // Extract JSON from response (handle markdown code blocks)
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error("Invalid response format");
      }

      const result = JSON.parse(jsonMatch[0]);
      setTrustScore(result.score);
      setAnalysis(result.analysis);

      toast({
        title: "Trust Score Calculated!",
        description: `Score: ${result.score}/5`,
      });
    } catch (error) {
      console.error("Error calculating trust score:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to calculate trust score. Please check your API key and try again.",
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
              <Award className="w-16 h-16 mx-auto mb-4 text-secondary" />
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                Trust <span className="bg-gradient-primary bg-clip-text text-transparent">Score</span>
              </h1>
              <p className="text-xl text-muted-foreground">
                Build your reputation through consistent contribution and quality help
              </p>
            </div>

            <div className="flex flex-col lg:flex-row gap-6">
              {/* Left Section - 70% */}
              <div className="lg:w-[70%] space-y-6">
                <Card className="border-2 border-primary">
                  <CardHeader>
                    <CardTitle className="text-2xl">What is Trust Score?</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-4">
                      Your Trust Score reflects your credibility and contribution to the bytee community. It's built through:
                    </p>
                    <ul className="space-y-2 text-muted-foreground">
                      <li className="flex items-start gap-2">
                        <span className="text-primary font-bold">•</span>
                        <span>Quality of help and mentorship provided</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary font-bold">•</span>
                        <span>Positive feedback from peers you've helped</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary font-bold">•</span>
                        <span>Consistency and reliability in commitments</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary font-bold">•</span>
                        <span>Active participation in community events</span>
                      </li>
                    </ul>
                  </CardContent>
                </Card>

                <div className="grid md:grid-cols-3 gap-6">
                  <Card>
                    <CardHeader>
                      <Star className="w-8 h-8 mb-2 text-secondary" />
                      <CardTitle>Earn Recognition</CardTitle>
                      <CardDescription>
                        Higher trust scores unlock badges, special privileges, and increase your visibility in the community.
                      </CardDescription>
                    </CardHeader>
                  </Card>

                  <Card>
                    <CardHeader>
                      <TrendingUp className="w-8 h-8 mb-2 text-secondary" />
                      <CardTitle>Track Progress</CardTitle>
                      <CardDescription>
                        Monitor your score growth over time and see how your contributions impact the community.
                      </CardDescription>
                    </CardHeader>
                  </Card>

                  <Card>
                    <CardHeader>
                      <Shield className="w-8 h-8 mb-2 text-secondary" />
                      <CardTitle>Build Trust</CardTitle>
                      <CardDescription>
                        A high trust score makes others more likely to seek your help and value your expertise.
                      </CardDescription>
                    </CardHeader>
                  </Card>
                </div>

                <Card className="bg-gradient-primary text-primary-foreground">
                  <CardHeader>
                    <CardTitle className="text-2xl">Trust Score Levels</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="bg-primary-foreground/10 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-semibold text-lg">🌱 Newcomer</h3>
                        <span className="text-sm">0-1.5 score</span>
                      </div>
                      <p className="text-sm opacity-90">Just getting started on your journey</p>
                    </div>

                    <div className="bg-primary-foreground/10 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-semibold text-lg">⭐ Contributor</h3>
                        <span className="text-sm">1.5-3.0 score</span>
                      </div>
                      <p className="text-sm opacity-90">Actively helping and building reputation</p>
                    </div>

                    <div className="bg-primary-foreground/10 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-semibold text-lg">💎 Expert</h3>
                        <span className="text-sm">3.0-4.0 score</span>
                      </div>
                      <p className="text-sm opacity-90">Recognized authority in specific areas</p>
                    </div>

                    <div className="bg-primary-foreground/10 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-semibold text-lg">👑 Community Leader</h3>
                        <span className="text-sm">4.0+ score</span>
                      </div>
                      <p className="text-sm opacity-90">Pillar of the bytee community</p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Right Section - 30% - Trust Score Calculator */}
              <div className="lg:w-[30%]">
                <Card className="sticky top-24">
                  <CardHeader>
                    <CardTitle>Check Trust Score</CardTitle>
                    <CardDescription>Enter reviews to calculate trust score using AI</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <form onSubmit={handleCalculateScore} className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="reviews">User Reviews & Feedback</Label>
                        <Textarea
                          id="reviews"
                          placeholder="Paste reviews and feedback about the user here..."
                          value={reviews}
                          onChange={(e) => setReviews(e.target.value)}
                          rows={8}
                          required
                        />
                        <p className="text-xs text-muted-foreground">
                          Enter multiple reviews separated by line breaks
                        </p>
                      </div>

                      <Button type="submit" className="w-full" variant="hero" disabled={isLoading}>
                        {isLoading ? "Calculating..." : "Calculate Trust Score"}
                      </Button>

                      {trustScore !== null && (
                        <div className="mt-6 space-y-4">
                          <div className="text-center p-6 bg-gradient-primary rounded-lg">
                            <div className="text-4xl font-bold text-primary-foreground mb-2">
                              {trustScore.toFixed(1)}/5.0
                            </div>
                            <div className="flex justify-center gap-1">
                              {[1, 2, 3, 4, 5].map((star) => (
                                <Star
                                  key={star}
                                  className={`w-5 h-5 ${
                                    star <= Math.round(trustScore)
                                      ? "fill-secondary text-secondary"
                                      : "text-primary-foreground/30"
                                  }`}
                                />
                              ))}
                            </div>
                          </div>
                          
                          {analysis && (
                            <Card>
                              <CardHeader>
                                <CardTitle className="text-sm">Analysis</CardTitle>
                              </CardHeader>
                              <CardContent>
                                <p className="text-sm text-muted-foreground">{analysis}</p>
                              </CardContent>
                            </Card>
                          )}
                        </div>
                      )}
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

export default TrustScore;
