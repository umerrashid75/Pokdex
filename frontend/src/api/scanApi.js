/**
 * scanApi.js – Axios bridge between the React frontend and FastAPI backend.
 * Sends a captured image blob to POST /v1/scan and returns the Pokedex JSON.
 */

import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || ''

/**
 * @param {Blob} imageBlob  – Raw image blob from the canvas capture
 * @returns {Promise<{label, display_name, type, confidence, dex_entry}>}
 */
export async function scanImage(imageBlob) {
    const formData = new FormData()
    formData.append('file', imageBlob, 'snapshot.jpg')

    const response = await axios.post(`${BASE_URL}/v1/scan`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30_000, // 30 s – model inference can be slow on first run
    })

    return response.data
}
