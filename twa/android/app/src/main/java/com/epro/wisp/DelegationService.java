package com.epro.wisp;

public class DelegationService extends
        com.google.androidbrowserhelper.trusted.DelegationService {
    @Override
    public void onCreate() {
        super.onCreate();
        registerExtraCommandHandler(new PlayBillingHandler(getApplicationContext()));
    }
}
