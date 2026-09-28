// =====================================================
// INTELLIVOICE - MULTILINGUAL VOICE ASSISTANT
// English | Tamil | Hindi
// =====================================================


// ================= HTML ELEMENTS =================

const camera = document.getElementById("camera");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const startButton = document.getElementById("startButton");
const detectionCanvas = document.getElementById("detectionCanvas");
const canvasContext = detectionCanvas.getContext("2d");

const languageScreen = document.getElementById("languageScreen");
const assistantScreen = document.getElementById("assistantScreen");


// ================= GLOBAL VARIABLES =================

let cameraStream = null;
let assistantActive = false;

let detectionTimer = null;
let reminderTimer = null;
let reminderStarted = false;

let lastAlert = "";
let lastAlertTime = 0;

let previousSizes = {};

const BACKEND_URL = "http://127.0.0.1:5000";


// ================= LANGUAGE SETTINGS =================

let selectedLanguage = null;

const languageVoices = {
    en: "en-US",
    ta: "ta-IN",
    hi: "hi-IN"
};


// ================= TRANSLATIONS =================

const translations = {

    en: {
        welcome: "Welcome to IntelliVoice. Please select your language. Say 1 for English, 2 for Tamil, or 3 for Hindi.",

        selected: "English selected.",

        start: "Double tap anywhere to start IntelliVoice.",

        stop: "Double tap anywhere to stop IntelliVoice.",

        activated: "IntelliVoice activated.",

        stopped: "IntelliVoice stopped.",

        waiting: "Waiting for detection...",

        scanning: "Scanning...",

        none: "None",

        person: "Person",
        car: "Car",
        bus: "Bus",
        bicycle: "Bicycle",
        motorbike: "Motorbike",

        left: "on your left",
        right: "on your right",
        ahead: "ahead",

        detected: "detected",
        approaching: "approaching",
        veryClose: "very close",

        cameraError: "Camera permission was denied or the camera is unavailable.",

        languageAgain: "Sorry, I did not understand. Please say 1 for English, 2 for Tamil, or 3 for Hindi."
    },


    ta: {
        welcome: "இன்டெல்லிவாய்ஸுக்கு வரவேற்கிறோம். மொழியைத் தேர்ந்தெடுக்கவும். ஆங்கிலத்திற்கு ஒன்று, தமிழுக்கு இரண்டு, இந்திக்கு மூன்று என்று சொல்லவும்.",

        selected: "தமிழ் தேர்ந்தெடுக்கப்பட்டது.",

        start: "இன்டெல்லிவாய்ஸை தொடங்க எங்கு வேண்டுமானாலும் இரண்டு முறை தட்டவும்.",

        stop: "இன்டெல்லிவாய்ஸை நிறுத்த எங்கு வேண்டுமானாலும் இரண்டு முறை தட்டவும்.",

        activated: "இன்டெல்லிவாய்ஸ் செயல்படத் தொடங்கியது.",

        stopped: "இன்டெல்லிவாய்ஸ் நிறுத்தப்பட்டது.",

        waiting: "கண்டறிவதற்காக காத்திருக்கிறது.",

        scanning: "சுற்றுப்புறத்தை ஆய்வு செய்கிறது.",

        none: "எதுவும் இல்லை",

        person: "ஒருவர்",
        car: "கார்",
        bus: "பேருந்து",
        bicycle: "மிதிவண்டி",
        motorbike: "மோட்டார் சைக்கிள்",

        left: "உங்கள் இடது புறத்தில்",
        right: "உங்கள் வலது புறத்தில்",
        ahead: "முன்னால்",

        detected: "கண்டறியப்பட்டது",
        approaching: "அருகில் வருகிறது",
        veryClose: "மிகவும் அருகில் உள்ளது",

        cameraError: "கேமரா அனுமதி மறுக்கப்பட்டது அல்லது கேமரா கிடைக்கவில்லை.",

        languageAgain: "மன்னிக்கவும், புரியவில்லை. ஆங்கிலத்திற்கு ஒன்று, தமிழுக்கு இரண்டு, இந்திக்கு மூன்று என்று சொல்லவும்."
    },


    hi: {
        welcome: "इंटेलीवॉइस में आपका स्वागत है। कृपया अपनी भाषा चुनें। अंग्रेज़ी के लिए एक, तमिल के लिए दो, या हिंदी के लिए तीन बोलें।",

        selected: "हिंदी चुनी गई है।",

        start: "इंटेलीवॉइस शुरू करने के लिए कहीं भी दो बार टैप करें।",

        stop: "इंटेलीवॉइस बंद करने के लिए कहीं भी दो बार टैप करें।",

        activated: "इंटेलीवॉइस शुरू हो गया है।",

        stopped: "इंटेलीवॉइस बंद हो गया है।",

        waiting: "पहचान का इंतज़ार है।",

        scanning: "जाँच की जा रही है।",

        none: "कुछ नहीं",

        person: "व्यक्ति",
        car: "कार",
        bus: "बस",
        bicycle: "साइकिल",
        motorbike: "मोटरसाइकिल",

        left: "आपकी बाईं ओर",
        right: "आपकी दाईं ओर",
        ahead: "आगे",

        detected: "दिखाई दिया",
        approaching: "पास आ रहा है",
        veryClose: "बहुत पास है",

        cameraError: "कैमरे की अनुमति नहीं मिली या कैमरा उपलब्ध नहीं है।",

        languageAgain: "माफ़ कीजिए, समझ नहीं आया। अंग्रेज़ी के लिए एक, तमिल के लिए दो, या हिंदी के लिए तीन बोलें।"
    }

};


// ================= TRANSLATION HELPERS =================

function getText(key) {

    const language = selectedLanguage || "en";

    return translations[language][key] || translations.en[key] || key;
}


function getObjectName(object) {

    if (object === "motorcycle") {
        object = "motorbike";
    }

    return getText(object) || object;
}


function getPosition(position) {

    if (position === "left") {
        return getText("left");
    }

    if (position === "right") {
        return getText("right");
    }

    return getText("ahead");
}


function makeDetectionMessage(object, position) {

    return `${getObjectName(object)} ${getPosition(position)} ${getText("detected")}.`;
}


function makeApproachMessage(object, position, veryClose) {

    const objectName = getObjectName(object);
    const direction = getPosition(position);

    if (veryClose) {
        return `${objectName} ${direction} ${getText("veryClose")}.`;
    }

    return `${objectName} ${direction} ${getText("approaching")}.`;
}


// ================= SPEECH QUEUE =================

let speechQueue = [];
let isSpeaking = false;


function speak(message, priority = 1) {

    if (!message) {
        return;
    }

    speechQueue.push({
        message: message,
        priority: priority
    });

    // Process higher-priority alerts first.
    speechQueue.sort((a, b) => b.priority - a.priority);

    processSpeechQueue();
}


function processSpeechQueue() {

    if (isSpeaking || speechQueue.length === 0) {
        return;
    }

    isSpeaking = true;

    const alert = speechQueue.shift();
    const speech = new SpeechSynthesisUtterance(alert.message);

    const languageCode = {
        en: "en-US",
        ta: "ta-IN",
        hi: "hi-IN"
    };

    const targetLanguage = languageCode[selectedLanguage] || "en-US";
    const voices = window.speechSynthesis.getVoices();

    speech.lang = targetLanguage;

    let matchingVoice = voices.find(voice =>
        voice.lang.toLowerCase() === targetLanguage.toLowerCase()
    );

    if (!matchingVoice) {
        const prefix = targetLanguage.split("-")[0].toLowerCase();

        matchingVoice = voices.find(voice =>
            voice.lang.toLowerCase().startsWith(prefix)
        );
    }

    if (matchingVoice) {
        speech.voice = matchingVoice;
        console.log("Using speech voice:", matchingVoice.name, matchingVoice.lang);
    } else {
        console.log("No matching voice found:", targetLanguage);
    }

    speech.rate = 0.9;
    speech.pitch = 1;
    speech.volume = 1;

    speech.onstart = () => {
        console.log("Speech started:", selectedLanguage, alert.message);
    };

    speech.onend = () => {
        console.log("Speech finished:", selectedLanguage);

        isSpeaking = false;
        processSpeechQueue();
    };

    speech.onerror = (event) => {
        console.error("Speech failed:", event.error);

        isSpeaking = false;
        processSpeechQueue();
    };

    window.speechSynthesis.speak(speech);

    console.log("Speech queued:", selectedLanguage, alert.message);
}


// ================= LANGUAGE SELECTION =================

function askLanguage() {

    window.speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(
        translations.en.welcome
    );

    speech.lang = "en-US";
    speech.rate = 0.9;
    speech.pitch = 1;
    speech.volume = 1;

    speech.onend = () => {

        startLanguageRecognition();
    };

    window.speechSynthesis.speak(speech);
}


function selectLanguage(language) {

    if (!translations[language]) {
        return;
    }

    selectedLanguage = language;

    // Stop any previous language prompt.
    window.speechSynthesis.cancel();

    // Clear old queued speech.
    speechQueue = [];
    isSpeaking = false;

    languageScreen.style.display = "none";
    assistantScreen.style.display = "block";

    // Reset alert history.
    lastAlert = "";
    lastAlertTime = 0;

    speak(getText("selected"), 2);

    setTimeout(() => {

        speak(getText("start"), 2);

    }, 1000);
}


// ================= LANGUAGE BUTTONS =================

document.querySelectorAll(".languageButton").forEach(button => {

    button.addEventListener("click", () => {

        const language = button.dataset.language;

        selectLanguage(language);
    });

});


// ================= VOICE LANGUAGE SELECTION =================

function startLanguageRecognition() {

    const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {

        console.log("Speech recognition is not supported in this browser.");

        return;
    }

    const recognition = new SpeechRecognition();

    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 3;

    recognition.onresult = function(event) {

        const answer = event.results[0][0].transcript
            .toLowerCase()
            .trim();

        console.log("Language selection heard:", answer);

        if (
            answer === "1" ||
            answer.includes("one") ||
            answer.includes("first") ||
            answer.includes("english")
        ) {

            selectLanguage("en");

        } else if (
            answer === "2" ||
            answer.includes("two") ||
            answer.includes("to") ||
            answer.includes("second") ||
            answer.includes("tamil")
        ) {

            selectLanguage("ta");

        } else if (
            answer === "3" ||
            answer.includes("three") ||
            answer.includes("free") ||
            answer.includes("third") ||
            answer.includes("hindi")
        ) {

            selectLanguage("hi");

        } else {

            speak(translations.en.languageAgain, 2);

            setTimeout(() => {

                startLanguageRecognition();

            }, 2500);
        }
    };

    recognition.onerror = function(event) {

        console.log("Speech recognition error:", event.error);
    };

    recognition.onend = function() {

        console.log("Language recognition ended.");
    };

    try {

        recognition.start();

        console.log("Listening for language selection...");

    } catch (error) {

        console.log("Could not start speech recognition:", error);
    }
}


// ================= STATUS UPDATE =================

function updateStatus(active) {

    const assistantDot = document.querySelector(
        ".status-item:nth-child(1) .dot"
    );

    const assistantText = document.querySelector(
        ".status-item:nth-child(1) span:last-child"
    );

    const voiceDot = document.querySelector(
        ".status-item:nth-child(2) .dot"
    );

    const voiceText = document.querySelector(
        ".status-item:nth-child(2) span:last-child"
    );

    if (assistantDot) {

        assistantDot.classList.toggle("green", active);
        assistantDot.classList.toggle("red", !active);
    }

    if (assistantText) {

        assistantText.textContent = active
            ? "ASSISTANT ACTIVE"
            : "ASSISTANT OFF";
    }

    if (voiceDot) {

        voiceDot.classList.toggle("green", active);
        voiceDot.classList.toggle("red", !active);
    }

    if (voiceText) {

        voiceText.textContent = active
            ? "VOICE ON"
            : "VOICE OFF";
    }
}


// ================= START CAMERA =================

async function startCamera() {

    if (!selectedLanguage) {
        return;
    }

    try {

        cameraStream = await navigator.mediaDevices.getUserMedia({

            video: true,
            audio: false

        });

        camera.srcObject = cameraStream;

        cameraPlaceholder.style.display = "none";

        assistantActive = true;

        startButton.textContent = getText("stop").toUpperCase();

        updateStatus(true);

        document.getElementById("lastAlert").textContent =
            getText("activated");

        document.getElementById("detectedObject").textContent =
            getText("scanning");

        speak(getText("activated"), 2);

        startDetection();

    } catch (error) {

        alert(getText("cameraError"));

        console.error(error);
    }
}


// ================= STOP CAMERA =================

function stopCamera() {

    canvasContext.clearRect(
        0,
        0,
        detectionCanvas.width,
        detectionCanvas.height
    );

    assistantActive = false;

    if (cameraStream) {

        cameraStream.getTracks().forEach(track => track.stop());

        cameraStream = null;
    }

    camera.srcObject = null;

    cameraPlaceholder.style.display = "block";

    startButton.textContent = getText("start").toUpperCase();

    updateStatus(false);

    document.getElementById("lastAlert").textContent =
        getText("stopped");

    document.getElementById("detectedObject").textContent =
        getText("none");

    previousSizes = {};

    stopDetection();

    // Stop queued warnings when the assistant is turned off.
    speechQueue = [];

    window.speechSynthesis.cancel();

    isSpeaking = false;

    speak(getText("stopped"), 2);
}


// ================= START DETECTION =================

function startDetection() {

    if (detectionTimer) {
        return;
    }

    detectionTimer = setInterval(() => {

        if (assistantActive) {

            sendFrameToAI();
        }

    }, 1500);
}


// ================= STOP DETECTION =================

function stopDetection() {

    if (detectionTimer) {

        clearInterval(detectionTimer);

        detectionTimer = null;
    }
}


// ================= SEND CAMERA FRAME TO AI =================

async function sendFrameToAI() {

    if (!cameraStream || !assistantActive) {
        return;
    }

    if (camera.readyState < 2) {
        return;
    }

    try {

        const canvas = document.createElement("canvas");

        canvas.width = camera.videoWidth;
        canvas.height = camera.videoHeight;

        detectionCanvas.width = camera.videoWidth;
        detectionCanvas.height = camera.videoHeight;

        const context = canvas.getContext("2d");

        context.drawImage(
            camera,
            0,
            0,
            canvas.width,
            canvas.height
        );

        const blob = await new Promise(resolve => {

            canvas.toBlob(
                resolve,
                "image/jpeg",
                0.7
            );

        });

        if (!blob) {
            return;
        }

        const formData = new FormData();

        formData.append(
            "frame",
            blob,
            "camera.jpg"
        );

        const response = await fetch(
            `${BACKEND_URL}/api/detect`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        processDetections(data.detections);

    } catch (error) {

        console.error("AI connection error:", error);
    }
}


// ================= PROCESS DETECTIONS =================

function processDetections(detections) {

    canvasContext.clearRect(
        0,
        0,
        detectionCanvas.width,
        detectionCanvas.height
    );

    if (!detections || detections.length === 0) {

        document.getElementById("detectedObject").textContent =
            getText("none");

        previousSizes = {};

        return;
    }

    // Priority of detected objects.
    const priorities = {

        bus: 5,
        car: 4,
        motorcycle: 3,
        bicycle: 3,
        person: 2

    };

    let approachingObjects = [];

    // Check every detected object.
    detections.forEach((detection) => {

        let objectName = detection.object;

        if (objectName === "motorcycle") {
            objectName = "motorbike";
        }

        const currentSize = detection.size_ratio;

        // Create history for this object.
        if (!previousSizes[objectName]) {

            previousSizes[objectName] = [];
        }

        previousSizes[objectName].push(currentSize);

        // Keep the latest 5 measurements.
        if (previousSizes[objectName].length > 5) {

            previousSizes[objectName].shift();
        }

        const history = previousSizes[objectName];

        // Need enough measurements before checking approach.
        if (history.length < 4) {
            return;
        }

        const oldSize = history[0];

        const newSize = history[history.length - 1];

        if (oldSize <= 0) {
            return;
        }

        const growth = (newSize - oldSize) / oldSize;

        const veryClose = newSize >= 0.25;

        // Object is approaching or very close.
        if (growth >= 0.20 || veryClose) {

            approachingObjects.push({

                object: objectName,

                position: detection.position,

                growth: growth,

                veryClose: veryClose,

                priority: priorities[detection.object] || 1

            });
        }
    });


    // ================= DRAW BOUNDING BOXES =================

    detections.forEach((detection) => {

        if (!detection.box) {
            return;
        }

        const box = detection.box;

        canvasContext.strokeStyle = "#00ff00";

        canvasContext.lineWidth = 4;

        canvasContext.strokeRect(

            box.x1,

            box.y1,

            box.x2 - box.x1,

            box.y2 - box.y1

        );

        canvasContext.fillStyle = "#00ff00";

        canvasContext.font = "bold 20px Arial";

        canvasContext.fillText(

            `${detection.object} ${Math.round(detection.confidence * 100)}%`,

            box.x1,

            Math.max(box.y1 - 8, 20)

        );
    });


    // ================= NO OBJECT APPROACHING =================

    if (approachingObjects.length === 0) {

        document.getElementById("detectedObject").textContent =

            detections.map(d => getObjectName(d.object)).join(", ");

        const detected = detections[0];

        const message = makeDetectionMessage(

            detected.object,

            detected.position

        );

        const currentTime = Date.now();

        if (

            message !== lastAlert ||

            currentTime - lastAlertTime > 5000

        ) {

            lastAlert = message;

            lastAlertTime = currentTime;

            document.getElementById("lastAlert").textContent = message;

            speak(message, 1);
        }

        return;
    }


    // ================= OBJECT APPROACHING =================

    // Sort by priority.
    approachingObjects.sort(

        (a, b) => b.priority - a.priority

    );

    // Select the highest-priority approaching object.
    const selected = approachingObjects[0];

    const objectName = selected.object;

    const position = selected.position;

    document.getElementById("detectedObject").textContent =

        `${getObjectName(objectName)} (${getPosition(position)})`;

    const message = makeApproachMessage(

        objectName,

        position,

        selected.veryClose

    );

    const currentTime = Date.now();

    // Prevent repeated alerts.
    if (

        message !== lastAlert ||

        currentTime - lastAlertTime > 5000

    ) {

        lastAlert = message;

        lastAlertTime = currentTime;

        document.getElementById("lastAlert").textContent = message;

        // Higher priority for approaching-object warnings.
        speak(message, 3);
    }
}


// ================= CAPITALIZE =================

function capitalize(word) {

    return word.charAt(0).toUpperCase() + word.slice(1);
}


// ================= REMINDER SYSTEM =================

function startReminder() {

    if (reminderStarted) {
        return;
    }

    reminderStarted = true;

    reminderTimer = setInterval(() => {

        if (!assistantActive && selectedLanguage) {

            speak(getText("start"), 1);
        }

    }, 30000);
}


// ================= DOUBLE-TAP ACTIVATION =================

document.addEventListener("dblclick", () => {

    // Do not activate the assistant before language selection.
    if (!selectedLanguage) {
        return;
    }

    if (assistantActive) {

        stopCamera();

    } else {

        startCamera();
    }
});


// ================= FIRST PAGE INTERACTION =================

document.addEventListener("click", () => {

    if (!selectedLanguage && !reminderStarted) {

        startLanguageRecognition();

    } else if (!assistantActive && selectedLanguage && !reminderStarted) {

        speak(getText("start"), 1);

        startReminder();
    }
});


// ================= PAGE LOAD =================

window.addEventListener("load", () => {

    // Show language selection first.
    languageScreen.style.display = "block";

    assistantScreen.style.display = "none";

    // Ask the user to select a language.
    askLanguage();
});
