window.addEventListener('load', function() {
    const audio = document.getElementById('audio');
    const labels = document.getElementById('labels');
    const predictions = document.getElementById('predictions').getElementsByTagName('li');

    for (const prediction of predictions) {
        const [startTime, endTime, language] = prediction.innerText.split(', ');
        const start = parseFloat(startTime);
        const end = parseFloat(endTime);

        const label = document.createElement('span');
        label.innerText = language;
        label.style.marginRight = '10px';
        label.style.backgroundColor = getRandomColor();
        labels.appendChild(label);

        audio.addEventListener('timeupdate', function() {
            const currentTime = audio.currentTime;
            if (currentTime >= start && currentTime <= end) {
                label.style.color = '#fff';
            } else {
                label.style.color = '#000';
            }
        });
    }

    function getRandomColor() {
        const letters = '0123456789ABCDEF';
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }
});
