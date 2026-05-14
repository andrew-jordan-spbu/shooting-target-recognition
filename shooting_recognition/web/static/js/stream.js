const shotsList  = document.getElementById('shots-list');
const totalScore = document.getElementById('total-score');

const evtSource = new EventSource('/events');

evtSource.onmessage = (e) => {
  const data = JSON.parse(e.data);

  if (data.type === 'shot') {
    const row = document.createElement('div');
    row.className = 'shot-row';
    row.innerHTML =
      `<span class="shot-num">Shot #${data.shot_number}</span>` +
      `<span class="shot-score score-${data.score}">${data.score} pts</span>`;
    shotsList.prepend(row);
    totalScore.textContent = data.total;
  }

  if (data.type === 'done') {
    evtSource.close();
  }
};

evtSource.onerror = () => evtSource.close();
