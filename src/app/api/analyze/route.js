import { NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Initialize Gemini — Securely using Environment Variables
const apiKey = process.env.GEMINI_API_KEY;
const genAI = apiKey ? new GoogleGenerativeAI(apiKey) : null;

export async function POST(request) {
  try {
    const body = await request.json();
    
    if (!genAI) {
      return NextResponse.json({ success: false, error: 'GEMINI_API_KEY is not configured on the server.' }, { status: 500 });
    }

    if (!body.image) {
      return NextResponse.json({ success: false, error: 'Missing "image" field.' }, { status: 400 });
    }

    const base64Data = body.image.includes(',') ? body.image.split(',')[1] : body.image;
    const mimeMatch = body.image.match(/^data:(image\/\w+);base64,/);
    const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg';
    
    const model = genAI.getGenerativeModel({ 
      model: "gemini-1.5-flash-latest",
      safetySettings: [
        { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" },
        { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_NONE" },
        { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_NONE" },
        { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_NONE" },
      ]
    });

    const prompt = `
      You are the CropGuard Expert Pathologist.
      
      Step 1: Check if the image contains a plant, leaf, or agricultural subject.
      If the image is of a person, animal, car, or any non-plant object, set "isNotPlant": true.
      
      Step 2: Return ONLY a JSON object:
      {
        "isNotPlant": boolean,
        "crop": "Species Name",
        "disease": "Disease Name",
        "isHealthy": boolean,
        "confidence": number (0.0 to 1.0),
        "severity": "Low/Medium/High",
        "advice": "Expert treatment protocol",
        "shouldSpray": boolean
      }
    `;

    const imageParts = [{ inlineData: { data: base64Data, mimeType } }];
    const result = await model.generateContent([prompt, ...imageParts]);
    const responseText = result.response.text();

    let aiData;
    try {
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      aiData = jsonMatch ? JSON.parse(jsonMatch[0]) : null;
    } catch (e) {
      console.error("Parse error", e);
    }

    if (!aiData) {
      aiData = {
        isNotPlant: responseText.toLowerCase().includes('not a plant'),
        crop: "Plant",
        disease: "Unknown",
        isHealthy: false,
        confidence: 0.5,
        severity: "Medium",
        advice: responseText,
        shouldSpray: false
      };
    }

    const formattedResult = {
      isNotPlant: !!aiData.isNotPlant,
      topPrediction: {
        label: aiData.disease || 'Unknown',
        confidence: aiData.confidence || 0.8,
        isHealthy: !!aiData.isHealthy,
        diseaseInfo: {
          crop: aiData.crop || 'Plant',
          disease: aiData.disease || 'Unknown',
          severity: aiData.severity || 'Medium',
          advice: aiData.advice || 'Standard monitoring protocol.'
        }
      },
      shouldSpray: !!aiData.shouldSpray,
      confidence: aiData.confidence || 0.8,
      isHealthy: !!aiData.isHealthy,
      timestamp: new Date().toISOString()
    };

    return NextResponse.json({ success: true, result: formattedResult });

  } catch (err) {
    console.error("Gemini API Error:", err);
    return NextResponse.json({ success: false, error: 'AI Engine Fault: ' + err.message }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({ status: 'ok', engine: 'Gemini 1.5 Flash' });
}
