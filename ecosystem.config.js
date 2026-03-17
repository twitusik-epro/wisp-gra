module.exports = {
  apps: [{
    name:         'wisp',
    script:       'server.js',
    cwd:          '/opt/gry/Wisp',
    instances:    1,
    autorestart:  true,
    watch:        false,
    max_memory_restart: '200M',
    env: {
      NODE_ENV: 'production',
    },
    error_file:  '/var/log/wisp/error.log',
    out_file:    '/var/log/wisp/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
  }]
};
