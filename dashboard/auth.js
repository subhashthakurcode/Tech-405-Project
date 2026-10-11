/**
 * Weather AI - Dummy Login & Sign Up Interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    if (window.lucide) {
        lucide.createIcons();
    }

    // Tab Elements
    const tabLogin = document.getElementById('tab-login');
    const tabSignup = document.getElementById('tab-signup');
    const formLogin = document.getElementById('form-login');
    const formSignup = document.getElementById('form-signup');
    const authTitle = document.getElementById('auth-title');
    const authSubtitle = document.getElementById('auth-subtitle');

    // Forms
    const loginEmail = document.getElementById('login-email');
    const loginPassword = document.getElementById('login-password');
    const btnFillDemo = document.getElementById('btn-fill-demo');
    const btnGoogle = document.getElementById('btn-google');
    const btnGithub = document.getElementById('btn-github');
    const linkForgot = document.getElementById('link-forgot');

    // Toast
    const toast = document.getElementById('toast-notice');
    const toastText = document.getElementById('toast-text');

    function showToast(msg, isSuccess = true, callback = null) {
        toastText.textContent = msg;
        toast.className = `toast-notice show ${isSuccess ? 'success' : ''}`;
        if (window.lucide) lucide.createIcons();

        setTimeout(() => {
            toast.classList.remove('show');
            if (callback) callback();
        }, 1200);
    }

    // Switch Tabs
    tabLogin.addEventListener('click', () => {
        tabLogin.classList.add('active');
        tabSignup.classList.remove('active');
        formLogin.classList.add('active');
        formSignup.classList.remove('active');
        authTitle.textContent = 'Welcome Back';
        authSubtitle.textContent = 'Sign in to your European Weather AI account';
    });

    tabSignup.addEventListener('click', () => {
        tabSignup.classList.add('active');
        tabLogin.classList.remove('active');
        formSignup.classList.add('active');
        formLogin.classList.remove('active');
        authTitle.textContent = 'Create Account';
        authSubtitle.textContent = 'Get access to 14-day LSTM climate predictions';
    });

    // Password Visibility Toggles
    document.querySelectorAll('.toggle-pwd').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-target');
            const input = document.getElementById(targetId);
            if (!input) return;

            if (input.type === 'password') {
                input.type = 'text';
                btn.innerHTML = '<i data-lucide="eye-off" style="width: 16px; height: 16px;"></i>';
            } else {
                input.type = 'password';
                btn.innerHTML = '<i data-lucide="eye" style="width: 16px; height: 16px;"></i>';
            }
            if (window.lucide) lucide.createIcons();
        });
    });

    // Auto-fill Demo
    if (btnFillDemo) {
        btnFillDemo.addEventListener('click', () => {
            loginEmail.value = 'subhash@weatherai.org';
            loginPassword.value = 'demo1234';
            showToast('Demo credentials populated!', true);
        });
    }

    // Handle Login Submit
    formLogin.addEventListener('submit', (e) => {
        e.preventDefault();
        const email = loginEmail.value.trim();
        const btnSubmit = document.getElementById('btn-login-submit');

        btnSubmit.disabled = true;
        btnSubmit.innerHTML = '<i data-lucide="loader" style="width:16px;height:16px;"></i> <span>Signing in...</span>';
        if (window.lucide) lucide.createIcons();

        setTimeout(() => {
            const userName = email.split('@')[0] || 'User';
            localStorage.setItem('weatherai_user', JSON.stringify({
                name: userName.charAt(0).toUpperCase() + userName.slice(1),
                email: email,
                loggedInAt: new Date().toISOString()
            }));

            showToast(`Welcome back, ${userName}! Redirecting...`, true, () => {
                window.location.href = 'index.html';
            });
        }, 600);
    });

    // Handle Sign Up Submit
    formSignup.addEventListener('submit', (e) => {
        e.preventDefault();
        const name = document.getElementById('signup-name').value.trim();
        const email = document.getElementById('signup-email').value.trim();
        const pwd = document.getElementById('signup-password').value;
        const confirmPwd = document.getElementById('signup-confirm-password').value;

        if (pwd !== confirmPwd) {
            showToast('Passwords do not match!', false);
            return;
        }

        const btnSubmit = document.getElementById('btn-signup-submit');
        btnSubmit.disabled = true;
        btnSubmit.innerHTML = '<i data-lucide="loader" style="width:16px;height:16px;"></i> <span>Creating account...</span>';
        if (window.lucide) lucide.createIcons();

        setTimeout(() => {
            localStorage.setItem('weatherai_user', JSON.stringify({
                name: name,
                email: email,
                loggedInAt: new Date().toISOString()
            }));

            showToast(`Account created for ${name}! Redirecting...`, true, () => {
                window.location.href = 'index.html';
            });
        }, 700);
    });

    // Social dummy logins
    [btnGoogle, btnGithub].forEach(btn => {
        if (!btn) return;
        btn.addEventListener('click', () => {
            const provider = btn.id === 'btn-google' ? 'Google' : 'GitHub';
            localStorage.setItem('weatherai_user', JSON.stringify({
                name: 'Subhash Thakur',
                email: `subhash@${provider.toLowerCase()}.com`,
                provider: provider,
                loggedInAt: new Date().toISOString()
            }));

            showToast(`Authenticated via ${provider}! Redirecting...`, true, () => {
                window.location.href = 'index.html';
            });
        });
    });

    if (linkForgot) {
        linkForgot.addEventListener('click', () => {
            showToast('Password reset link sent to registered email!', true);
        });
    }
});
