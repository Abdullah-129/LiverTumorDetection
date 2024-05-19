 <h1>Liver Tumor Detection Model</h1>
    <p>This model was trained using Generative Adversarial Networks (GANs) to detect liver tumors from CT scans.</p>

    <h2>How to Run</h2>
    <ol>
        <li><strong>Navigate to the Server Folder:</strong><br>
            <code>cd server</code></li>
        <li><strong>Run the Server:</strong><br>
            <code>python server.py</code></li>
        <li><strong>Access the Model:</strong><br>
            The server should now be running and accessible. You can interact with the model using API requests.</li>
    </ol>

    <h2>API Endpoints</h2>
    <ul>
        <li><code>/detect-tumor</code>: Use this endpoint to send a CT scan image and receive a prediction for the presence of a liver tumor.</li>
    </ul>
