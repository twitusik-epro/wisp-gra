package com.epro.wisp;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.browser.trusted.TrustedWebActivityCallbackRemote;

import com.android.billingclient.api.AcknowledgePurchaseParams;
import com.android.billingclient.api.BillingClient;
import com.android.billingclient.api.BillingClientStateListener;
import com.android.billingclient.api.BillingResult;
import com.android.billingclient.api.ConsumeParams;
import com.android.billingclient.api.PendingPurchasesParams;
import com.android.billingclient.api.ProductDetails;
import com.android.billingclient.api.Purchase;
import com.android.billingclient.api.PurchaseHistoryRecord;
import com.android.billingclient.api.PurchasesUpdatedListener;
import com.android.billingclient.api.QueryProductDetailsParams;
import com.android.billingclient.api.QueryPurchaseHistoryParams;
import com.android.billingclient.api.QueryPurchasesParams;
import com.google.androidbrowserhelper.trusted.ExtraCommandHandler;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class PlayBillingHandler implements ExtraCommandHandler, PurchasesUpdatedListener {

    private static final String TAG = "PlayBillingHandler";

    // Outer bundle keys
    private static final String KEY_SUCCESS = "success";
    private static final String KEY_VERSION = "digital_goods_api_version";
    private static final int    DG_API_VER  = 2;

    // Command names
    private static final String CMD_GET_DETAILS        = "getDetails";
    private static final String CMD_ACKNOWLEDGE        = "acknowledge";
    private static final String CMD_CONSUME            = "consume";
    private static final String CMD_LIST_PURCHASES     = "listPurchases";
    private static final String CMD_LIST_PURCHASE_HIST = "listPurchaseHistory";

    // getDetails protocol keys
    private static final String PARAM_ITEM_IDS  = "getDetails.itemIds";
    private static final String RESP_GET        = "getDetails.response";
    private static final String RESP_GET_CODE   = "getDetails.responseCode";
    private static final String RESP_GET_LIST   = "getDetails.detailsList";

    // acknowledge
    private static final String PARAM_ACK_TOKEN = "acknowledge.purchaseToken";
    private static final String RESP_ACK        = "acknowledge.response";
    private static final String RESP_ACK_CODE   = "acknowledge.responseCode";

    // consume
    private static final String PARAM_CONSUME_TOKEN = "consume.purchaseToken";
    private static final String RESP_CONSUME        = "consume.response";
    private static final String RESP_CONSUME_CODE   = "consume.responseCode";

    // listPurchases
    private static final String RESP_LIST       = "listPurchases.response";
    private static final String RESP_LIST_CODE  = "listPurchases.responseCode";
    private static final String RESP_LIST_ITEMS = "listPurchases.purchasesList";

    // listPurchaseHistory
    private static final String RESP_HIST       = "listPurchaseHistory.response";
    private static final String RESP_HIST_CODE  = "listPurchaseHistory.responseCode";
    private static final String RESP_HIST_ITEMS = "listPurchaseHistory.purchasesList";

    // ItemDetails bundle keys – protocol from androidbrowserhelper:billing (decompiled)
    private static final String ITEM_ID             = "itemDetails.id";
    private static final String ITEM_TITLE          = "itemDetails.title";
    private static final String ITEM_DESC           = "itemDetails.description";
    private static final String ITEM_CURRENCY       = "itemDetails.currency";
    private static final String ITEM_VALUE          = "itemDetails.value";
    private static final String ITEM_TYPE           = "itemDetails.type";
    private static final String ITEM_ICON_URL       = "itemDetails.url";
    private static final String ITEM_SUBS_PERIOD    = "itemDetails.subsPeriod";
    private static final String ITEM_FREE_TRIAL     = "itemDetails.freeTrialPeriod";
    private static final String ITEM_INTRO_PERIOD   = "itemDetails.introPricePeriod";
    private static final String ITEM_INTRO_CURRENCY = "itemDetails.introPriceCurrency";
    private static final String ITEM_INTRO_VALUE    = "itemDetails.introPriceValue";
    private static final String ITEM_INTRO_CYCLES   = "itemDetails.introPriceCycles";

    // PurchaseDetails bundle keys
    private static final String PD_ITEM_ID = "purchaseDetails.itemId";
    private static final String PD_TOKEN   = "purchaseDetails.purchaseToken";

    // Chromium response codes
    private static final int CR_OK           = 0;
    private static final int CR_ERROR        = 1;
    private static final int CR_ALREADY_OWNED = 2;
    private static final int CR_NOT_OWNED    = 3;
    private static final int CR_UNAVAILABLE  = 4;

    private final Context mContext;
    private BillingClient mBillingClient;

    public PlayBillingHandler(Context context) {
        mContext = context.getApplicationContext();
        // BillingClient init deferred until first real command (not mock)
    }

    private void ensureBillingClient() {
        if (mBillingClient != null) return;
        mBillingClient = BillingClient.newBuilder(mContext)
                .setListener(this)
                .enablePendingPurchases(
                        PendingPurchasesParams.newBuilder()
                                .enableOneTimeProducts()
                                .build())
                .build();
        mBillingClient.startConnection(new BillingClientStateListener() {
            @Override
            public void onBillingSetupFinished(BillingResult r) {
                Log.d(TAG, "BillingClient ready: " + r.getResponseCode());
            }
            @Override
            public void onBillingServiceDisconnected() {
                Log.w(TAG, "BillingClient disconnected");
                mBillingClient = null;
            }
        });
    }

    @Override
    public void onPurchasesUpdated(BillingResult r, @Nullable List<Purchase> purchases) {}

    // -------------------------------------------------------------------------
    // ExtraCommandHandler
    // -------------------------------------------------------------------------

    @Override
    @Nullable
    public Bundle handleExtraCommand(Context context,
                                     String commandName,
                                     Bundle args,
                                     @Nullable TrustedWebActivityCallbackRemote callback) {
        Log.d(TAG, "handleExtraCommand: " + commandName);

        if (!isKnown(commandName)) return null;

        ensureBillingClient();
        Bundle ack = ack();
        if (callback == null) return ack;
        if (mBillingClient != null && mBillingClient.isReady()) {
            dispatch(commandName, args, callback);
        } else {
            connectThen(commandName, args, callback);
        }
        return ack;
    }

    private boolean isKnown(String cmd) {
        return CMD_GET_DETAILS.equals(cmd)
                || CMD_ACKNOWLEDGE.equals(cmd)
                || CMD_CONSUME.equals(cmd)
                || CMD_LIST_PURCHASES.equals(cmd)
                || CMD_LIST_PURCHASE_HIST.equals(cmd);
    }

    private void connectThen(String cmd, Bundle args, TrustedWebActivityCallbackRemote cb) {
        if (mBillingClient == null) ensureBillingClient();
        mBillingClient.startConnection(new BillingClientStateListener() {
            @Override
            public void onBillingSetupFinished(BillingResult r) {
                if (r.getResponseCode() == BillingClient.BillingResponseCode.OK) {
                    dispatch(cmd, args, cb);
                } else {
                    send(cb, cmd, errorResponse(cmd));
                }
            }
            @Override
            public void onBillingServiceDisconnected() {}
        });
    }

    private void dispatch(String cmd, Bundle args, TrustedWebActivityCallbackRemote cb) {
        switch (cmd) {
            case CMD_GET_DETAILS:        getDetailsReal(args, cb); break;
            case CMD_ACKNOWLEDGE:        acknowledge(args, cb);    break;
            case CMD_CONSUME:            consume(args, cb);        break;
            case CMD_LIST_PURCHASES:     listPurchases(cb);        break;
            case CMD_LIST_PURCHASE_HIST: listPurchaseHistory(cb);  break;
        }
    }

    // --- acknowledge ---
    private void acknowledge(Bundle args, TrustedWebActivityCallbackRemote cb) {
        String token = args.getString(PARAM_ACK_TOKEN);
        if (token == null) { send(cb, CMD_ACKNOWLEDGE, errorResponse(CMD_ACKNOWLEDGE)); return; }
        mBillingClient.acknowledgePurchase(
                AcknowledgePurchaseParams.newBuilder().setPurchaseToken(token).build(),
                br -> {
                    Bundle o = ack(); Bundle r = new Bundle();
                    r.putInt(RESP_ACK_CODE, toCr(br.getResponseCode()));
                    o.putBundle(RESP_ACK, r);
                    send(cb, CMD_ACKNOWLEDGE, o);
                });
    }

    // --- consume ---
    private void consume(Bundle args, TrustedWebActivityCallbackRemote cb) {
        String token = args.getString(PARAM_CONSUME_TOKEN);
        if (token == null) { send(cb, CMD_CONSUME, errorResponse(CMD_CONSUME)); return; }
        mBillingClient.consumeAsync(
                ConsumeParams.newBuilder().setPurchaseToken(token).build(),
                (br, t) -> {
                    Bundle o = ack(); Bundle r = new Bundle();
                    r.putInt(RESP_CONSUME_CODE, toCr(br.getResponseCode()));
                    o.putBundle(RESP_CONSUME, r);
                    send(cb, CMD_CONSUME, o);
                });
    }

    // --- listPurchases ---
    private void listPurchases(TrustedWebActivityCallbackRemote cb) {
        ensureBillingClient();
        mBillingClient.queryPurchasesAsync(
                QueryPurchasesParams.newBuilder()
                        .setProductType(BillingClient.ProductType.INAPP).build(),
                (br, list) -> {
                    Bundle o = ack(); Bundle r = new Bundle();
                    if (br.getResponseCode() == BillingClient.BillingResponseCode.OK) {
                        r.putInt(RESP_LIST_CODE, CR_OK);
                        android.os.Parcelable[] arr = new android.os.Parcelable[list.size()];
                        for (int i = 0; i < list.size(); i++) arr[i] = toPdBundle(list.get(i));
                        r.putParcelableArray(RESP_LIST_ITEMS, arr);
                    } else { r.putInt(RESP_LIST_CODE, toCr(br.getResponseCode())); }
                    o.putBundle(RESP_LIST, r);
                    send(cb, CMD_LIST_PURCHASES, o);
                });
    }

    // --- listPurchaseHistory ---
    private void listPurchaseHistory(TrustedWebActivityCallbackRemote cb) {
        ensureBillingClient();
        mBillingClient.queryPurchaseHistoryAsync(
                QueryPurchaseHistoryParams.newBuilder()
                        .setProductType(BillingClient.ProductType.INAPP).build(),
                (br, records) -> {
                    Bundle o = ack(); Bundle r = new Bundle();
                    if (br.getResponseCode() == BillingClient.BillingResponseCode.OK && records != null) {
                        r.putInt(RESP_HIST_CODE, CR_OK);
                        android.os.Parcelable[] arr = new android.os.Parcelable[records.size()];
                        for (int i = 0; i < records.size(); i++) arr[i] = toPdBundle(records.get(i));
                        r.putParcelableArray(RESP_HIST_ITEMS, arr);
                    } else { r.putInt(RESP_HIST_CODE, toCr(br.getResponseCode())); }
                    o.putBundle(RESP_HIST, r);
                    send(cb, CMD_LIST_PURCHASE_HIST, o);
                });
    }

    // --- Real getDetails (BillingClient) ---
    private void getDetailsReal(Bundle args, TrustedWebActivityCallbackRemote cb) {
        String[] ids = args.getStringArray(PARAM_ITEM_IDS);
        if (ids == null || ids.length == 0) { send(cb, CMD_GET_DETAILS, errorResponse(CMD_GET_DETAILS)); return; }
        List<QueryProductDetailsParams.Product> products = new ArrayList<>();
        for (String id : ids) {
            products.add(QueryProductDetailsParams.Product.newBuilder()
                    .setProductId(id).setProductType(BillingClient.ProductType.INAPP).build());
        }
        mBillingClient.queryProductDetailsAsync(
                QueryProductDetailsParams.newBuilder().setProductList(products).build(),
                (br, list) -> {
                    Bundle o = ack(); Bundle r = new Bundle();
                    if (br.getResponseCode() == BillingClient.BillingResponseCode.OK) {
                        r.putInt(RESP_GET_CODE, CR_OK);
                        android.os.Parcelable[] arr = new android.os.Parcelable[list.size()];
                        for (int i = 0; i < list.size(); i++) arr[i] = toItemBundle(list.get(i));
                        r.putParcelableArray(RESP_GET_LIST, arr);
                    } else { r.putInt(RESP_GET_CODE, toCr(br.getResponseCode())); }
                    o.putBundle(RESP_GET, r);
                    send(cb, CMD_GET_DETAILS, o);
                });
    }

    private Bundle toItemBundle(ProductDetails pd) {
        Bundle b = new Bundle();
        b.putString(ITEM_ID,           pd.getProductId());
        b.putString(ITEM_TITLE,        pd.getTitle());
        b.putString(ITEM_DESC,         pd.getDescription());
        b.putString(ITEM_TYPE,         "inapp");
        b.putString(ITEM_ICON_URL,     "");
        b.putString(ITEM_SUBS_PERIOD,  "");
        b.putString(ITEM_FREE_TRIAL,   "");
        b.putString(ITEM_INTRO_PERIOD, "");
        b.putString(ITEM_INTRO_CURRENCY, "");
        b.putString(ITEM_INTRO_VALUE,  "");
        b.putInt(ITEM_INTRO_CYCLES,    0);
        ProductDetails.OneTimePurchaseOfferDetails offer = pd.getOneTimePurchaseOfferDetails();
        if (offer != null) {
            b.putString(ITEM_CURRENCY, offer.getPriceCurrencyCode());
            b.putString(ITEM_VALUE,    microsToDecimal(offer.getPriceAmountMicros()));
        } else {
            b.putString(ITEM_CURRENCY, "");
            b.putString(ITEM_VALUE,    "0");
        }
        return b;
    }

    private static String microsToDecimal(long micros) {
        long cents = micros / 10000L;
        return String.format(Locale.US, "%d.%02d", cents / 100, cents % 100);
    }

    private Bundle toPdBundle(Purchase p) {
        Bundle b = new Bundle();
        List<String> products = p.getProducts();
        b.putString(PD_ITEM_ID, products.isEmpty() ? "" : products.get(0));
        b.putString(PD_TOKEN,   p.getPurchaseToken());
        return b;
    }

    private Bundle toPdBundle(PurchaseHistoryRecord r) {
        Bundle b = new Bundle();
        List<String> products = r.getProducts();
        b.putString(PD_ITEM_ID, products.isEmpty() ? "" : products.get(0));
        b.putString(PD_TOKEN,   r.getPurchaseToken());
        return b;
    }

    private Bundle ack() {
        Bundle b = new Bundle();
        b.putBoolean(KEY_SUCCESS, true);
        b.putInt(KEY_VERSION, DG_API_VER);
        return b;
    }

    private Bundle errorResponse(String cmd) {
        Bundle o = ack(); Bundle r = new Bundle();
        switch (cmd) {
            case CMD_GET_DETAILS:        r.putInt(RESP_GET_CODE,     CR_ERROR); o.putBundle(RESP_GET,     r); break;
            case CMD_ACKNOWLEDGE:        r.putInt(RESP_ACK_CODE,     CR_ERROR); o.putBundle(RESP_ACK,     r); break;
            case CMD_CONSUME:            r.putInt(RESP_CONSUME_CODE, CR_ERROR); o.putBundle(RESP_CONSUME, r); break;
            case CMD_LIST_PURCHASES:     r.putInt(RESP_LIST_CODE,    CR_ERROR); o.putBundle(RESP_LIST,    r); break;
            case CMD_LIST_PURCHASE_HIST: r.putInt(RESP_HIST_CODE,    CR_ERROR); o.putBundle(RESP_HIST,    r); break;
        }
        return o;
    }

    private static int toCr(int code) {
        switch (code) {
            case BillingClient.BillingResponseCode.OK:                  return CR_OK;
            case BillingClient.BillingResponseCode.ITEM_ALREADY_OWNED: return CR_ALREADY_OWNED;
            case BillingClient.BillingResponseCode.ITEM_NOT_OWNED:     return CR_NOT_OWNED;
            case BillingClient.BillingResponseCode.ITEM_UNAVAILABLE:   return CR_UNAVAILABLE;
            default:                                                     return CR_ERROR;
        }
    }

    private void send(TrustedWebActivityCallbackRemote cb, String cmd, Bundle bundle) {
        try {
            cb.runExtraCallback(cmd, bundle);
        } catch (Exception e) {
            Log.e(TAG, "send callback failed for " + cmd + ": " + e);
        }
    }
}
