function updateTaskbar(username) {
    const userMenu = document.getElementById('user-menu');
    const signinPrompt = document.getElementById('signin-prompt');
    const usernameDisplay = document.getElementById('username');
    const signoutBtn = document.getElementById('signout-btn');
    const showSigninBtn = document.getElementById('show-signin-btn');

    if (username) {
        signinPrompt.classList.add('hidden');
        userMenu.classList.remove('hidden');
        usernameDisplay.textContent = `Hey, ${username}! 🐝`;
    } else {
        userMenu.classList.add('hidden');
        signinPrompt.classList.remove('hidden');
    }

    if (signoutBtn) {
        signoutBtn.addEventListener('click', () => {
            if (window.confirm('Are you sure you want to sign out?')) {
                localStorage.removeItem('user');
                document.cookie = 'username=; path=/; max-age=0';
                window.location.href = '/signin';
            }
        });
    }

    if (showSigninBtn) {
        showSigninBtn.addEventListener('click', () => {
            window.location.href = '/signin';
        });
    }
}

// Initialize taskbar on page load
document.addEventListener('DOMContentLoaded', () => {
    const user = localStorage.getItem('user');
    updateTaskbar(user);
});