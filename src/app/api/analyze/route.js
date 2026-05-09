/**
 * Hardware API Endpoint — POST /api/analyze
 * ============================================
 * Accepts a base64-encoded image and returns disease analysis.
 * 
 * This endpoint exists so that:
 *   1. Future hardware devices (Raspberry Pi, Arduino, ESP32)
 *      can POST images directly from a connected camera module.
 *   2. Third-party integrations can call this API programmatically.
 * 
 * Request body:
 *   { "image": "data:image/jpeg;base64,..." }
 * 
 * Response:
 *   { "success": true, "result": { light, shouldSpray, predictions, ... } }
 */

import { NextResponse } from 'next/server';
import { CLASS_NAMES, CLASS_LABELS, HEALTHY_CLASSES, DISEASE_INFO, SPRAY_CONFIG } from '@/lib/constants';

export async function POST(request) {
  try {
    const body = await request.json();
    
    if (!body.image) {
      return NextResponse.json(
        { success: false, error: 'Missing "image" field. Send a base64-encoded image.' },
        { status: 400 }
      );
    }

    // For now, return demo results (same logic as client-side demo mode).
    // When the real model is integrated server-side (e.g., via ONNX Runtime),
    // replace this with actual inference.
    
    const demoResults = getDemoResult();

    return NextResponse.json({
      success: true,
      result: demoResults,
      hardware_note: 'This endpoint is ready for hardware integration. POST a base64 image to get analysis.',
    });

  } catch (err) {
    return NextResponse.json(
      { success: false, error: err.message },
      { status: 500 }
    );
  }
}

// Health check
export async function GET() {
  return NextResponse.json({
    status: 'ok',
    model_ready: false, // Will be true when ONNX Runtime is set up server-side
    supported_classes: CLASS_NAMES.length,
    version: '1.0.0',
    usage: 'POST /api/analyze with { "image": "data:image/jpeg;base64,..." }',
  });
}

function getDemoResult() {
  const scenarios = [
    { className: 'Apple___healthy', isHealthy: true },
    { className: 'Apple___Apple_scab', isHealthy: false },
    { className: 'Tomato___Late_blight', isHealthy: false },
    { className: 'Grape___Black_rot', isHealthy: false },
  ];
  const pick = scenarios[Math.floor(Math.random() * scenarios.length)];
  const confidence = 0.85 + Math.random() * 0.12;
  const isHealthy = pick.isHealthy;

  let light = 'green';
  if (!isHealthy && confidence >= SPRAY_CONFIG.sprayConfidenceThreshold) light = 'red';
  else if (!isHealthy) light = 'amber';

  return {
    topPrediction: {
      className: pick.className,
      label: CLASS_LABELS[pick.className],
      confidence,
      isHealthy,
      diseaseInfo: DISEASE_INFO[pick.className] || null,
    },
    light,
    shouldSpray: light === 'red',
    isDemo: true,
    timestamp: new Date().toISOString(),
  };
}
