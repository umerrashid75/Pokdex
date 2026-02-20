/**
 * CameraView.jsx
 * Streams the rear camera to a <video> element inside the Pokédex screen.
 * Exposes a captureFrame() ref method that snapshots the current frame as a Blob.
 */

import { useEffect, useRef, useImperativeHandle, forwardRef, useState } from 'react'

const CameraView = forwardRef(function CameraView({ isScanning }, ref) {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const [camError, setCamError] = useState(null)

    // Start the camera stream
    useEffect(() => {
        let stream = null

        const startCamera = async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: { ideal: 'environment' }, // rear camera on phones
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                    },
                    audio: false,
                })
                if (videoRef.current) {
                    videoRef.current.srcObject = stream
                }
            } catch (err) {
                console.error('Camera error:', err)
                setCamError('Camera permission denied or not available.')
            }
        }

        startCamera()

        return () => {
            // Cleanup: stop all tracks when component unmounts
            if (stream) stream.getTracks().forEach(t => t.stop())
        }
    }, [])

    /**
     * captureFrame() – draw current video frame to an off-screen canvas,
     * then return it as a JPEG Blob suitable for upload.
     */
    useImperativeHandle(ref, () => ({
        captureFrame: () =>
            new Promise((resolve, reject) => {
                const video = videoRef.current
                const canvas = canvasRef.current
                if (!video || !canvas) return reject(new Error('Video not ready'))

                canvas.width = video.videoWidth || 640
                canvas.height = video.videoHeight || 480

                const ctx = canvas.getContext('2d')
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

                canvas.toBlob(
                    blob => (blob ? resolve(blob) : reject(new Error('Canvas toBlob failed'))),
                    'image/jpeg',
                    0.92,
                )
            }),
    }))

    if (camError) {
        return (
            <div className="cam-error">
                <span>⚠</span>
                <p>{camError}</p>
                <small>Enable camera access and refresh the page.</small>
            </div>
        )
    }

    return (
        <div className="camera-container">
            {/* Scanline overlay for the "dot matrix" effect */}
            <div className="scanline-overlay" />

            {/* Scanning reticle crosshair */}
            {isScanning && <div className="scan-reticle" />}

            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="camera-feed"
            />

            {/* Hidden canvas used for frame capture */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>
    )
})

export default CameraView
