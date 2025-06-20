function monitorProgress() {
  interval = setInterval(() => {
    console.log("Fetching progress for:", APP_NAME);
    fetch(`/progress/${APP_NAME}`)
      .then(...)
      .catch(...);
  }, 2000);
}

function startTraining() {
  fetch(`/lda/${APP_NAME}`, { method: "POST" })
    .then(res => {
      console.log("Training request sent to:", `/lda/${APP_NAME}`, "status:", res.status);
      return res.json();
    })
    .then(data => {
      if (data.status === "started") monitorProgress();
    })
    .catch(...);
}
